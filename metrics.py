# -*- coding: utf-8 -*-
"""
Detection metrics (per-image TP/FP accumulation)
- mAP: per-image TP/FP, compute() does numpy aggregation only
- F1: computed alongside mAP via ap_per_class
- Confusion matrix: optional, final epoch only
"""
import numpy as np
import torch
import torchvision


# -----------------------------------------------------
# numpy ap_per_class
# -----------------------------------------------------

def _compute_ap(recall, precision):
    """101-point interp AP (COCO standard)"""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    try:
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    except AttributeError:
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def _smooth(y, f=0.05):
    """Box filter"""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute per-class AP (10 IoU thresholds, COCO standard)
    tp:       (N, 10) bool
    conf:     (N,) float
    pred_cls: (N,) int
    target_cls: (M,) int
    Returns: precision, recall, f1, ap (per-class), map50, map50_95
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    if nc == 0:
        return 0.0, 0.0, 0.0, np.array([]), 0.0, 0.0

    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        ci_mask = pred_cls == c
        n_l = nt[ci]
        n_p = ci_mask.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[ci_mask]).cumsum(0)
        tpc = tp[ci_mask].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[ci_mask], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[ci_mask], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = _compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i = _smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]

    map50 = ap[:, 0].mean() if ap.shape[1] > 0 else 0.0
    map50_95 = ap.mean()
    return p.mean(), r.mean(), f1.mean(), ap, map50, map50_95


# -----------------------------------------------------
# Main metrics class: per-image TP/FP accumulation
# -----------------------------------------------------

class AdvancedDetMetrics:
    """
    DetMetrics: per-image TP/FP, compute() does numpy aggregation.
    mAP@0.5:0.95 (10 IoU thresholds, COCO standard)

    Note: filtering is done in validation_step (conf=0.001), not here.
    """
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

    def update(self, preds, targets):
        """Per-image TP/FP computation, store as numpy"""
        for pred, target in zip(preds, targets):
            p_boxes = pred['boxes']
            p_scores = pred['scores']
            p_labels = pred['labels']
            t_boxes = target['boxes']
            t_labels = target['labels']

            if p_boxes.shape[0] == 0:
                self.stats['tp'].append(np.zeros((0, 10), dtype=bool))
                self.stats['conf'].append(np.zeros(0))
                self.stats['pred_cls'].append(np.zeros(0))
                self.stats['target_cls'].append(t_labels.cpu().numpy())
                continue

            if t_boxes.shape[0] == 0:
                self.stats['tp'].append(np.zeros((p_boxes.shape[0], 10), dtype=bool))
                self.stats['conf'].append(p_scores.cpu().numpy())
                self.stats['pred_cls'].append(p_labels.cpu().numpy())
                self.stats['target_cls'].append(np.zeros(0))
                continue

            iou = torchvision.ops.box_iou(t_boxes, p_boxes)  # (M, N)
            correct_class = t_labels[:, None] == p_labels  # (M, N)
            iou_matched = (iou * correct_class).cpu().numpy()

            # Greedy assignment in descending confidence order, as in the COCO protocol
            conf_order = np.argsort(-p_scores.cpu().numpy())
            correct = np.zeros((p_boxes.shape[0], 10), dtype=bool)
            for ti, threshold in enumerate(self.iouv.tolist()):
                matched_gt = set()
                for pi in conf_order:
                    # Ground truth of the same class with IoU >= threshold
                    gt_candidates = np.where(iou_matched[:, pi] >= threshold)[0]
                    if len(gt_candidates) == 0:
                        continue
                    # Among those, take the one with the highest IoU that is still free
                    best_gt = gt_candidates[np.argmax(iou_matched[gt_candidates, pi])]
                    if best_gt not in matched_gt:
                        correct[pi, ti] = True
                        matched_gt.add(best_gt)

            self.stats['tp'].append(correct)
            self.stats['conf'].append(p_scores.cpu().numpy())
            self.stats['pred_cls'].append(p_labels.cpu().numpy())
            self.stats['target_cls'].append(t_labels.cpu().numpy())

    def compute(self):
        """Aggregate per-image stats, compute mAP/precision/recall/F1"""
        if not self.stats['tp']:
            return {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0,
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        tp = np.concatenate(self.stats['tp'], 0)
        conf = np.concatenate(self.stats['conf'], 0)
        pred_cls = np.concatenate(self.stats['pred_cls'], 0)
        target_cls = np.concatenate(self.stats['target_cls'], 0)

        if tp.shape[0] == 0:
            return {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0,
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        p, r, f1, ap, map50, map50_95 = ap_per_class(tp, conf, pred_cls, target_cls)

        map_75 = float(ap[:, 5].mean()) if ap.ndim == 2 and ap.shape[1] > 5 else 0.0
        return {'map': float(map50_95), 'map_50': float(map50), 'map_75': float(map_75),
                'precision': float(p), 'recall': float(r), 'f1': float(f1)}

    def compute_confusion_matrix(self, preds, targets):
        """
        Call only at final epoch.
        Returns (num_classes+1, num_classes+1) numpy array.
        """
        n = self.num_classes + 1  # +1 for background
        cm = np.zeros((n, n), dtype=np.int64)
        for pred, target in zip(preds, targets):
            keep = pred['scores'] > 0.05
            p_boxes, p_labels, p_scores = pred['boxes'][keep], pred['labels'][keep], pred['scores'][keep]
            t_boxes, t_labels = target['boxes'], target['labels']

            if t_boxes.shape[0] == 0:
                for l in p_labels.cpu().numpy():
                    cm[l, self.num_classes] += 1
                continue
            if p_boxes.shape[0] == 0:
                for l in t_labels.cpu().numpy():
                    cm[self.num_classes, l] += 1
                continue

            iou = torchvision.ops.box_iou(p_boxes, t_boxes)
            # Greedy assignment in descending confidence order; each ground truth can be
            # matched at most once, following the COCO evaluation protocol
            sorted_idx = p_scores.argsort(descending=True)
            matched_gt = set()
            assigned_pred = {}
            for pi in sorted_idx:
                best_iou, best_gi = iou[pi].max(0)
                best_gi = best_gi.item()
                if best_iou > self.iou_threshold and best_gi not in matched_gt:
                    assigned_pred[pi.item()] = best_gi
                    matched_gt.add(best_gi)
            for pi in range(len(p_boxes)):
                if pi in assigned_pred:
                    cm[p_labels[pi].item(), t_labels[assigned_pred[pi]].item()] += 1
                else:
                    cm[p_labels[pi].item(), self.num_classes] += 1
            for gi in range(len(t_boxes)):
                if gi not in matched_gt:
                    cm[self.num_classes, t_labels[gi].item()] += 1
        return cm

    def reset(self):
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
