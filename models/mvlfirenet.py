# -*- coding: utf-8 -*-
"""
MVLFireNet: a lightweight forest fire and smoke detector with multi-granularity
vision-language enhancement.

Data flow::

    input 640x640
      -> Backbone  ELAN-style CNN + SPPF            -> C3, C4, C5
      -> Neck      FPN + MSA at P5 + CMF at P4/P3   -> P3, P4, P5
      -> Head      RT-DETR decoder, 300 queries      -> boxes + classes
      -> MVLE      dual-pathway alignment (training only, dropped at inference)
"""
import math

import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from models import RTDETRDecoder
from models.modules import Conv, ELANBlock, SPPF
from models.neck import FPNNeck

import config
from loss import HungarianMatcher, SetCriterion
from metrics import AdvancedDetMetrics
from utils import box_cxcywh_to_xyxy


class Backbone(nn.Module):
    """ELAN-style CNN with SPPF at the deepest stage.

    Returns feature maps at strides 8, 16 and 32 with 128, 128 and 256 channels
    by default (see ``config.BACKBONE_OUT_CHANNELS``).
    """

    def __init__(self, out_channels=(128, 128, 256)):
        super().__init__()
        c3, c4, c5 = out_channels
        self.stem = Conv(3, 16, k=3, s=2)
        self.layer2 = Conv(16, 32, k=3, s=2)
        self.layer2_block = ELANBlock(32, 64, e=0.5)

        self.layer3_conv = Conv(64, 64, k=3, s=2)
        self.layer3_block = ELANBlock(64, c3, e=0.5)

        self.layer4_conv = Conv(c3, c4, k=3, s=2)
        self.layer4_block = ELANBlock(c4, c4, enhance=True)

        self.layer5_conv = Conv(c4, c5, k=3, s=2)
        self.layer5_block = ELANBlock(c5, c5, enhance=True)

        self.sppf = SPPF(c5, c5, k=5, shortcut=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)

        c3 = self.layer3_block(self.layer3_conv(self.layer2_block(x)))
        c4 = self.layer4_block(self.layer4_conv(c3))
        c5 = self.sppf(self.layer5_block(self.layer5_conv(c4)))
        return c3, c4, c5


class MVLFireNet(pl.LightningModule):
    """Lightweight fire and smoke detector.

    Args:
        num_classes: number of foreground classes (2: fire, smoke).
        use_mvle: attach the MVLE semantic alignment branch. Enabled during
            training and disabled at inference, where the branch would contribute
            nothing to the predictions.

    Note:
        The box and classification heads are zero-initialised on their last layer
        because they predict residual corrections on top of anchors; a non-zero
        init breaks that residual path and destabilises early training.
    """

    def __init__(self, num_classes=None, use_mvle=False):
        super().__init__()
        self.num_classes = num_classes or config.NUM_CLASSES
        self.use_mvle = use_mvle

        self.backbone = Backbone(tuple(config.BACKBONE_OUT_CHANNELS))
        self.neck = FPNNeck(
            tuple(config.BACKBONE_OUT_CHANNELS),
            tuple(config.NECK_OUT_CHANNELS),
            use_msa=config.USE_MSA,
            use_cmf=config.USE_CMF,
        )

        if use_mvle:
            from models.mvle import MVLEBranch
            self.mvle = MVLEBranch(
                clip_model_path=config.CLIP_MODEL_PATH,
                sa_dim_global=config.MVLE_DIM_GLOBAL,
                sa_dim_local=config.MVLE_DIM_LOCAL,
                use_pos_embed=config.MVLE_USE_POS_EMBED,
                temperature=config.MVLE_TEMPERATURE,
                temperature_local=config.MVLE_TEMPERATURE_LOCAL,
                text_dim=config.TEXT_ENCODER_DIM,
            )

        self.head = RTDETRDecoder(
            num_classes=self.num_classes,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_queries=config.DECODER_NUM_QUERIES,
            num_decoder_layers=config.DECODER_NUM_LAYERS,
            nhead=config.DECODER_NHEAD,
            in_channels=config.NECK_OUT_CHANNELS,
            num_denoising=config.NUM_DENOISING,
            cls_noise_ratio=config.CLS_NOISE_RATIO,
            box_noise_scale=config.BOX_NOISE_SCALE,
        )

        matcher = HungarianMatcher(
            cost_class=config.COST_CLASS,
            cost_bbox=config.COST_BBOX,
            cost_giou=config.COST_GIOU,
            cost_nwd=config.COST_NWD,
            use_focal_loss=True,
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
        )
        weight_dict = dict(config.LOSS_WEIGHTS)
        for i in range(config.DECODER_NUM_LAYERS):
            for k, v in config.LOSS_WEIGHTS.items():
                if 'clip' not in k and '_dn' not in k:
                    weight_dict[f'{k}_{i}'] = v
        for k, v in config.LOSS_WEIGHTS.items():
            if 'clip' not in k and '_dn' not in k:
                weight_dict[f'{k}_dn'] = v

        self.criterion = SetCriterion(
            self.num_classes, matcher, weight_dict,
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            eos_coef=config.EOS_COEF,
            use_uni_set=config.USE_UNI_SET,
        )
        self.metrics = AdvancedDetMetrics(num_classes=self.num_classes, iou_threshold=0.5)
        self._nan_count = 0

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        p3, p4, p5 = self.neck(feats)
        out = self.head([p3, p4, p5], targets=targets)
        # MVLE aligns against the backbone features, not the neck outputs.
        out['_backbone_feats'] = feats
        return out

    def post_process(self, outputs):
        return outputs['pred_logits'].sigmoid()

    # -- training --

    def training_step(self, batch, batch_idx):
        outputs = self(batch['images'], targets=batch['targets'])
        loss_dict, _ = self.criterion(outputs, batch['targets'], epoch=self.current_epoch)

        loss_dict.update(self._mvle_losses(batch, outputs))

        weight_dict = self.criterion.weight_dict
        total = sum(loss_dict.get(k, 0.0) * weight_dict.get(k, 1.0)
                    for k in loss_dict if not k.startswith('_'))

        if torch.isnan(total) or torch.isinf(total):
            self._nan_count += 1
            self.log('nan_count', self._nan_count)
            if self._nan_count >= 10:
                raise RuntimeError(f'aborting: {self._nan_count} consecutive NaN/Inf losses')
            bad = [k for k, v in loss_dict.items()
                   if isinstance(v, torch.Tensor) and (torch.isnan(v) or torch.isinf(v))]
            print(f'[NaN] epoch={self.current_epoch}, step={self.global_step}, keys={bad}')
            return None
        self._nan_count = 0

        self.log('train_loss', total, prog_bar=True, batch_size=config.BATCH_SIZE)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1 and not k.startswith('_'):
                self.log(k, v, prog_bar=True, batch_size=config.BATCH_SIZE)
        return total

    def _mvle_losses(self, batch, outputs):
        """Compute the MVLE alignment losses, gated by epoch range and weight.

        The global pathway runs on non-Mosaic samples only: a Mosaic canvas
        combines four unrelated scenes, so its scene-level caption no longer
        describes the image. The local pathway uses per-box captions that stay
        valid because they are tracked through the Mosaic transform.
        """
        losses = {}
        weight_dict = self.criterion.weight_dict
        w_global = weight_dict.get('loss_clip_global', 0.0)
        w_local = weight_dict.get('loss_clip_local', 0.0)

        in_window = (config.MVLE_START_EPOCH <= self.current_epoch < config.MVLE_END_EPOCH)
        if not (self.use_mvle and in_window and (w_global > 0 or w_local > 0)):
            return losses

        c3, c4, c5 = outputs['_backbone_feats']
        is_mosaic = batch.get('is_mosaic') or [False] * c5.shape[0]

        if w_global > 0:
            captions = batch.get('global_captions')
            if captions:
                keep = [i for i, (m, c) in enumerate(zip(is_mosaic, captions))
                        if not m and c is not None]
                if keep:
                    losses['loss_clip_global'] = self.mvle.global_loss(
                        c5[keep], [captions[i] for i in keep])

        if w_local > 0:
            local = batch.get('local_captions')
            if local:
                losses['loss_clip_local'] = self.mvle.local_loss(c5, local)

        # Ramp the alignment loss in over the first few epochs: the projection
        # layers start random and a full-strength loss early on only adds noise.
        warmup = config.MVLE_WARMUP_EPOCHS
        if warmup > 0:
            step = self.current_epoch - config.MVLE_START_EPOCH
            if step < warmup:
                scale = step / warmup
                for k in losses:
                    losses[k] = losses[k] * scale
        return losses

    # -- validation --

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['images'])
        targets = batch['targets']

        loss_dict, _ = self.criterion(outputs, targets, epoch=self.current_epoch,
                                     main_only=True)
        loss = sum(loss_dict.get(k, 0.0) * self.criterion.weight_dict.get(k, 1.0)
                   for k in loss_dict if not k.startswith('_'))
        self.log('val_loss', loss, batch_size=config.BATCH_SIZE, sync_dist=True)

        preds, targets_out = [], []
        probas = self.post_process(outputs)
        pred_boxes = outputs['pred_boxes']
        for i, tgt in enumerate(targets):
            h, w = tgt.get('orig_size', (640, 640))
            cls_scores, cls_labels = probas[i].max(-1)
            keep = cls_scores > config.EVAL_CONF_THRESHOLD

            pred_boxes_i = box_cxcywh_to_xyxy(pred_boxes[i])
            pred_boxes_i[:, 0::2] *= w
            pred_boxes_i[:, 1::2] *= h
            tgt_boxes = box_cxcywh_to_xyxy(tgt['boxes'])
            tgt_boxes[:, 0::2] *= w
            tgt_boxes[:, 1::2] *= h

            preds.append({'boxes': pred_boxes_i[keep],
                          'scores': cls_scores[keep],
                          'labels': cls_labels[keep]})
            targets_out.append({'boxes': tgt_boxes, 'labels': tgt['labels']})

        self.metrics.update(preds, targets_out)
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            self._val_raw_preds.extend(preds)
            self._val_raw_targets.extend(targets_out)
        return loss

    def on_validation_epoch_start(self):
        self._val_raw_preds = []
        self._val_raw_targets = []

    def on_validation_epoch_end(self):
        if self.trainer.world_size > 1:
            self._gather_stats()
        result = self.metrics.compute() if self.trainer.global_rank == 0 else None

        if self.trainer.world_size > 1:
            result = self._broadcast(result)

        self.log('val_map', result['map'], prog_bar=True, sync_dist=True)
        self.log('val_map_50', result['map_50'], prog_bar=True, sync_dist=True)
        self.log('val_map_75', result['map_75'], sync_dist=True)
        self.log('val_precision', result['precision'], sync_dist=True)
        self.log('val_recall', result['recall'], sync_dist=True)
        self.log('val_f1', result['f1'], sync_dist=True)

        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            if self.trainer.global_rank == 0:
                print('[VAL] Confusion Matrix:\n'
                      + str(self.metrics.compute_confusion_matrix(
                          self._val_raw_preds, self._val_raw_targets)))
            self._val_raw_preds.clear()
            self._val_raw_targets.clear()
        self.metrics.reset()

    def _broadcast(self, result):
        import torch.distributed as dist
        box = [result]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _gather_stats(self):
        """Merge validation statistics across ranks onto rank 0."""
        import torch.distributed as dist
        rank = self.trainer.global_rank
        if rank == 0:
            gathered = [None] * self.trainer.world_size
            dist.gather_object(self.metrics.stats, gathered, dst=0)
            merged = {k: [] for k in self.metrics.stats}
            for shard in gathered:
                for k in merged:
                    merged[k].extend(shard[k])
            self.metrics.stats = merged

            if self.trainer.current_epoch == self.trainer.max_epochs - 1:
                preds = [None] * self.trainer.world_size
                tgts = [None] * self.trainer.world_size
                dist.gather_object(self._val_raw_preds, preds, dst=0)
                dist.gather_object(self._val_raw_targets, tgts, dst=0)
                self._val_raw_preds = [p for shard in preds for p in shard]
                self._val_raw_targets = [t for shard in tgts for t in shard]
        else:
            dist.gather_object(self.metrics.stats, None, dst=0)
            if self.trainer.current_epoch == self.trainer.max_epochs - 1:
                dist.gather_object(self._val_raw_preds, None, dst=0)
                dist.gather_object(self._val_raw_targets, None, dst=0)

    # -- optimisation --

    def configure_optimizers(self):
        """Build the AdamW parameter groups and the warmup/flat/cosine schedule.

        Three logical groups are kept apart so each can carry its own learning
        rate: head+neck, backbone, and the MVLE projection layers. Each group is
        further split into decay / no-decay by weight name and parent module type
        (biases and normalisation parameters do not decay).
        """
        bn_types = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        decay, no_decay = [[], []], [[], []]   # index 0: weights, 1: proj/MVLE
        decay_bn, decay_proj, nodecay_bn, nodecay_proj = [], [], [], []
        modules = dict(self.named_modules())

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Match on prefix, never by substring: "head" appears inside module
            # names such as global_pool.query_pos_head and would be misclassified.
            is_proj = 'mvle' in name
            parent = modules.get('.'.join(name.split('.')[:-1]), None)
            no_wd = 'bias' in name or isinstance(parent, bn_types)

            if is_proj:
                (nodecay_proj if no_wd else decay_proj).append(param)
            else:
                (no_decay[0] if no_wd else decay[0]).append(param)

        wd = config.WEIGHT_DECAY
        accumulate = getattr(self.trainer, 'accumulate_grad_batches', 1) if self.trainer else 1
        world_size = self.trainer.world_size if self.trainer else 1
        # Scale weight decay so the effective value does not depend on the
        # GPU count or the accumulation factor.
        effective_wd = wd * config.BATCH_SIZE * world_size * accumulate / config.NBS

        param_groups = [
            {'params': decay[0], 'lr': config.LR, 'weight_decay': effective_wd},
            {'params': no_decay[0], 'lr': config.LR, 'weight_decay': 0.0},
            {'params': decay_proj, 'lr': config.LR_MVLE, 'weight_decay': effective_wd},
            {'params': nodecay_proj, 'lr': config.LR_MVLE, 'weight_decay': 0.0},
        ]
        optimizer = optim.AdamW(param_groups, betas=(config.MOMENTUM, 0.999), eps=1e-8)

        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1000
        max_epochs = self.trainer.max_epochs if self.trainer else 200
        steps_per_epoch = total_steps // max_epochs
        warmup_steps = max(1, int(config.WARMUP_EPOCHS * steps_per_epoch))
        flat_steps = int(config.FLAT_EPOCHS * steps_per_epoch)
        gamma = config.LR_GAMMA

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps - 1, 1)
            if config.FLAT_EPOCHS > 0 and step < flat_steps:
                return 1.0
            decay_start = max(warmup_steps, flat_steps)
            progress = (step - decay_start) / max(1, total_steps - decay_start)
            return gamma + (1.0 - gamma) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}