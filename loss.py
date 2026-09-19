# -*- coding: utf-8 -*-
"""
Losses and matching for MVLFireNet.

The detector is trained with a set-prediction objective:

* **MAL** (:class:`SetCriterion.loss_labels_mal`) for classification, where the
  target score of a positive query is ``IoU ** gamma`` and the positive weight is
  the hard label. This compresses target scores and smooths the gradient
  contribution of hard samples, which matters because fire and smoke are
  amorphous and only coarsely annotated.
* **L1 + GIoU** for box regression, with GIoU weighted above L1 because overlap
  quality matters more than exact coordinates on such boundaries.
* **NWD** as an extra matching cost only. It has no loss term: optimising it
  directly was measured to lower the mAP@0.5 ceiling, since it targets
  coordinate precision while the evaluation metric rewards overlap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


def calculate_nwd(box1, box2, eps=1e-7):
    """Normalized Wasserstein distance between two sets of cxcywh boxes."""
    b1_cx, b1_cy, b1_w, b1_h = box1.unbind(-1)
    b2_cx, b2_cy, b2_w, b2_h = box2.unbind(-1)

    # Broadcast the first set against the second to build an NxM matrix.
    b1_cx = b1_cx.unsqueeze(1)
    b1_cy = b1_cy.unsqueeze(1)
    b1_w = b1_w.unsqueeze(1)
    b1_h = b1_h.unsqueeze(1)

    b2_cx = b2_cx.unsqueeze(0)
    b2_cy = b2_cy.unsqueeze(0)
    b2_w = b2_w.unsqueeze(0)
    b2_h = b2_h.unsqueeze(0)

    # Squared Wasserstein distance between the box centres and sizes.
    w2_sq = (b1_cx - b2_cx).pow(2) + \
            (b1_cy - b2_cy).pow(2) + \
            ((b1_w - b2_w) / 2.0).pow(2) + \
            ((b1_h - b2_h) / 2.0).pow(2)

    C = 0.5
    return torch.exp(-torch.sqrt(w2_sq + eps) / C)


def pairwise_nwd(box1, box2, eps=1e-7):
    """Element-wise NWD for two aligned sets of ``(N, 4)`` cxcywh boxes.

    Unlike :func:`calculate_nwd` this does not broadcast, so it applies to boxes
    that are already paired by the matcher.
    """
    w2 = (box1[:, 0] - box2[:, 0]).pow(2) + \
         (box1[:, 1] - box2[:, 1]).pow(2) + \
         ((box1[:, 2] - box2[:, 2]) / 2.0).pow(2) + \
         ((box1[:, 3] - box2[:, 3]) / 2.0).pow(2)
    return torch.exp(-torch.sqrt(w2 + eps) / 0.5)


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truth.

    Cost combines classification, L1, GIoU and NWD terms. The NWD term is
    included purely to help pair small, weakly localised boxes; it is not used as
    a training loss.
    """

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                 cost_nwd=2.0, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class; self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou; self.cost_nwd = cost_nwd
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha; self.gamma = gamma

    @torch.no_grad()
    @torch.compiler.disable
    def forward(self, outputs, targets, epoch=0):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64))
                    for _ in range(bs)]

        if self.use_focal_loss:
            out_prob_cls = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob_cls ** self.gamma) * (-(1 - out_prob_cls + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob_cls) ** self.gamma) * (-(out_prob_cls + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_nwd = 1.0 - calculate_nwd(out_bbox, tgt_bbox)

        C = (self.cost_bbox * cost_bbox
             + self.cost_class * cost_class
             + self.cost_giou * cost_giou
             + self.cost_nwd * cost_nwd)

        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=100.0, posinf=100.0, neginf=-100.0)
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
    """Match denoising queries back to the ground-truth boxes they were derived from."""
    dn_match_indices = []
    dn_match_indices = []
    idx_groups = torch.as_tensor([0] + gt_groups[:-1]).cumsum_(0)
    for i, num_gt in enumerate(gt_groups):
        if num_gt > 0:
            gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
            gt_idx = gt_idx.repeat(dn_num_group)
            dn_match_indices.append((dn_pos_idx[i], gt_idx))
        else:
            dn_match_indices.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
    return dn_match_indices


class SetCriterion(nn.Module):
    """Computes the detection losses.

    Args:
        num_classes: number of foreground classes.
        matcher: :class:`HungarianMatcher` instance.
        weight_dict: maps loss names to weights, including the per-layer and
            denoising variants.
        alpha: weight applied to the negative term of the MAL loss.
        gamma: exponent applied to the IoU target score in MAL.
        eos_coef: weight of the background class in the cross-entropy fallback.
        use_uni_set: share box losses across decoder layers using the union of the
            matched indices, which stabilises training of the later layers.
    """

    def __init__(self, num_classes, matcher, weight_dict, alpha=0.25, gamma=2.0,
                 eos_coef=0.1, use_uni_set=True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = alpha; self.gamma = gamma
        self.eos_coef = eos_coef
        self.use_uni_set = use_uni_set
        self.register_buffer('empty_weight', torch.ones(self.num_classes))

    def loss_labels(self, outputs, targets, indices, num_boxes, values=None):
        """Classification loss: MAL when focal-style matching is in use."""
        if self.matcher.use_focal_loss:
            return self.loss_labels_mal(outputs, targets, indices, num_boxes, values=values)

        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  self.empty_weight, ignore_index=self.num_classes)
        return {'loss_ce': loss_ce}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """Modified Adaptive Label loss.

        The target score of a matched query is ``IoU ** gamma`` rather than a hard
        1.0, which compresses scores and smooths the gradient weight of hard
        samples. Positive queries are weighted by the hard label; ``alpha`` only
        scales the negative term, unlike focal loss where it scales the positive.

        Args:
            values: precomputed IoU values from the matcher, if available.
        """
        assert 'pred_boxes' in outputs
        src_logits = outputs['pred_logits']
        nq = src_logits.shape[1]
        idx = self._get_src_permutation_idx(indices)

        if values is None:
            src_boxes = outputs['pred_boxes'][idx].float()
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).float()
            with torch.no_grad():
                iou_mat, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
                ious = iou_mat.diag().clamp(0, 1).nan_to_num(0.0)
        else:
            ious = values

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        target_score = target_score.pow(self.gamma)
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits.float(), target_score.float(),
                                                   weight=weight.float(), reduction='none')
        loss = loss.mean(1).sum() * nq / max(num_boxes, 1)
        return {'loss_ce': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1, GIoU and NWD terms for the matched boxes."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].float()
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).float()

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                      box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.clamp(min=-2.0, max=2.0).sum() / num_boxes
        loss_nwd = (1.0 - pairwise_nwd(src_boxes, target_boxes)).sum() / num_boxes
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou, 'loss_nwd': loss_nwd}

    def _get_go_indices(self, indices, indices_aux_list):
        """Union of the matched indices across all decoder layers.

        Each ground-truth box keeps only one assignment per batch: the query that
        matched it in the largest number of layers. This keeps a single ground
        truth from being supervised by several different queries at once.
        """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                       for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_index(match_indices):
        """Split match indices into ``((batch_idx, src_idx), dst_idx)``."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def get_loss(self, outputs, targets, indices, num_boxes, loss_type='cls', values=None):
        if loss_type == 'cls':
            return self.loss_labels(outputs, targets, indices, num_boxes, values=values)
        elif loss_type == 'boxes':
            return self.loss_boxes(outputs, targets, indices, num_boxes)
        return {}

    @torch.compiler.disable
    def forward(self, outputs, targets, epoch=0, main_only=False, compute_det_losses=True):
        """Compute the loss dictionary for one forward pass.

        Args:
            main_only: only evaluate the main output, skipping the auxiliary
                decoder layers. Used during validation.
            compute_det_losses: set to False to run only the matcher and return
                the indices without building the loss terms.

        Returns:
            ``(losses, indices)``. The indices are returned even when
            ``compute_det_losses`` is False so callers can reuse the matching.
        """
        # The matcher always runs: the MVLE local loss reuses its indices.
        outputs_for_match = {k: v for k, v in outputs.items()
                             if k not in ['aux_outputs', 'dn_meta', 'dn_pred_logits', 'dn_pred_boxes']}
        indices = self.matcher(outputs_for_match, targets, epoch=epoch)

        if not compute_det_losses:
            return {}, indices

        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        # Union matching: box losses share one set of indices across layers.
        indices_go = None
        num_boxes_go = num_boxes
        if self.use_uni_set and not main_only and 'aux_outputs' in outputs:
            indices_aux_list = [self.matcher(aux, targets, epoch=epoch)
                                for aux in outputs['aux_outputs']]
            indices_go = self._get_go_indices(indices, indices_aux_list)
            num_boxes_go = max(sum(len(x[0]) for x in indices_go), 1)

        # Classification uses the plain per-layer matching; box regression uses
        # the union set.
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        if indices_go is not None:
            losses.update(self.loss_boxes(outputs, targets, indices_go, num_boxes_go))
        else:
            losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        if not main_only:
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices_i = self.matcher(aux_outputs, targets, epoch=epoch)
                    l_dict = self.loss_labels(aux_outputs, targets, indices_i, num_boxes)
                    l_dict.update(self.loss_boxes(
                        aux_outputs, targets,
                        indices_go if indices_go is not None else indices_i,
                        num_boxes_go if indices_go is not None else num_boxes))
                    for k, v in l_dict.items():
                        losses[f'{k}_{i}'] = v

            if 'dn_meta' in outputs and outputs['dn_meta'] is not None:
                dn_meta = outputs['dn_meta']
                dn_bboxes = outputs['dn_bboxes']
                dn_scores = outputs['dn_scores']

                dn_match_indices = get_dn_match_indices(
                    dn_meta['dn_pos_idx'], dn_meta['dn_num_group'],
                    [len(t["labels"]) for t in targets])
                dn_idx_tuple, dn_gt_idx = self._get_index(dn_match_indices)

                dn_losses = {}
                gt_cls = torch.cat([t["labels"] for t in targets])
                gt_bboxes = torch.cat([t["boxes"] for t in targets])
                dn_num_matched = max(num_boxes * dn_meta['dn_num_group'], 1)

                dn_bs, dn_nq = dn_scores.shape[:2]
                dn_onehot = torch.zeros((dn_bs, dn_nq, self.num_classes),
                                        device=dn_scores.device)
                dn_targets = torch.full((dn_bs, dn_nq), self.num_classes,
                                        device=dn_scores.device, dtype=gt_cls.dtype)
                dn_targets[dn_idx_tuple] = gt_cls[dn_gt_idx]
                dn_onehot[dn_idx_tuple[0], dn_idx_tuple[1], gt_cls[dn_gt_idx]] = 1.0

                # MAL on the denoising queries.
                dn_pred_matched = dn_bboxes[dn_idx_tuple].float()
                dn_gt_matched = gt_bboxes[dn_gt_idx].float()
                with torch.no_grad():
                    iou_mat, _ = box_iou(box_cxcywh_to_xyxy(dn_pred_matched),
                                         box_cxcywh_to_xyxy(dn_gt_matched))
                    dn_ious = iou_mat.diag().clamp(0, 1).nan_to_num(0.0)
                dn_target_score_o = torch.zeros_like(dn_targets, dtype=dn_scores.dtype)
                dn_target_score_o[dn_idx_tuple] = dn_ious.to(dn_target_score_o.dtype)
                dn_target_score = dn_target_score_o.unsqueeze(-1) * dn_onehot
                dn_pred_score = dn_scores.sigmoid().detach()
                dn_target_score = dn_target_score.pow(self.gamma)
                dn_weight = (self.alpha * dn_pred_score.pow(self.gamma) * (1 - dn_onehot)
                             + dn_onehot)
                ce = F.binary_cross_entropy_with_logits(
                    dn_scores.float(), dn_target_score.float(),
                    weight=dn_weight.float(), reduction='none')
                dn_losses['loss_ce'] = ce.mean(1).sum() * dn_nq / dn_num_matched

                if len(dn_gt_idx):
                    dn_pred_b = dn_bboxes[dn_idx_tuple].float()
                    dn_gt_b = gt_bboxes[dn_gt_idx].float()
                    dn_losses['loss_bbox'] = (F.l1_loss(dn_pred_b, dn_gt_b, reduction='sum')
                                              / dn_num_matched)
                    giou = 1 - torch.diag(generalized_box_iou(
                        box_cxcywh_to_xyxy(dn_pred_b), box_cxcywh_to_xyxy(dn_gt_b)))
                    dn_losses['loss_giou'] = giou.clamp(min=-2.0, max=2.0).sum() / dn_num_matched
                    dn_losses['loss_nwd'] = ((1.0 - pairwise_nwd(dn_pred_b, dn_gt_b)).sum()
                                             / dn_num_matched)
                else:
                    _zero = dn_scores.sum() * 0
                    dn_losses['loss_bbox'] = _zero
                    dn_losses['loss_giou'] = _zero
                    dn_losses['loss_nwd'] = _zero

                dn_losses_out = {}
                for k, v in dn_losses.items():
                    dn_losses_out[k + '_dn'] = v
                losses.update(dn_losses_out)

        return losses, indices
