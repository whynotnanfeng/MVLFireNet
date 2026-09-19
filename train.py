# -*- coding: utf-8 -*-
"""
Training entry point for MVLFireNet.

Two augmentations are controlled independently through ``config.py``:

* ``MOSAIC_START`` / ``MOSAIC_END`` -- epoch window for Mosaic augmentation.
* ``MVLE_START_EPOCH`` / ``MVLE_END_EPOCH`` -- epoch window for the semantic
  alignment losses.
"""
import math
import os
import random
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

import config
from data.datasets import VLDataModule
from models.mvlfirenet import MVLFireNet

torch.set_float32_matmul_precision('medium')


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        os.environ.pop("PYTHONHASHSEED", None)


class MosaicControl(pl.Callback):
    """Turns Mosaic augmentation on and off across epochs.

    Mosaic is disabled for the final epochs so the model can converge on
    unaugmented images.
    """

    def __init__(self):
        self._active = None

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        datamodule = trainer.datamodule
        should_enable = config.MOSAIC_START <= epoch < config.MOSAIC_END
        if self._active == should_enable:
            return
        if datamodule is not None and hasattr(datamodule, 'train_dataset'):
            datamodule.train_dataset.close_mosaic = not should_enable
            datamodule._train_loader = None
            if trainer.global_rank == 0:
                state = "on" if should_enable else "off"
                print(f"[Mosaic] epoch {epoch}: Mosaic {state}")
        self._active = should_enable


class ModelEMA(pl.Callback):
    """Exponential moving average of the trainable weights, held on CPU.

    BN running statistics are averaged alongside the parameters so that a
    checkpoint contains a consistent model: mixing EMA weights with training-time
    BN buffers produces different results at evaluation time.
    """

    def __init__(self, decay=0.9999, tau=1000):
        super().__init__()
        self.decay = decay
        self.tau = tau
        self.ema_state_dict = {}
        self.steps = 0
        self._trainable_params = []
        self._buffers = []
        self._buffer_state = {}
        self._last_requires_grad = None

    def on_fit_start(self, trainer, pl_module):
        self._trainable_params = [(n, p) for n, p in pl_module.named_parameters()
                                  if p.requires_grad]
        self._last_requires_grad = {n: p.requires_grad
                                    for n, p in pl_module.named_parameters()}
        self.ema_state_dict = {n: v.detach().clone().cpu()
                               for n, v in self._trainable_params}
        self._buffers = list(pl_module.named_buffers())
        self._buffer_state = {n: b.detach().clone().cpu() for n, b in self._buffers}

    def _refresh_trainable_params(self, pl_module, rank=0):
        """Track parameters whose requires_grad changed since the last epoch."""
        current = {n: p.requires_grad for n, p in pl_module.named_parameters()}
        if self._last_requires_grad is None or current == self._last_requires_grad:
            return
        newly_trainable = [n for n, p in pl_module.named_parameters()
                           if p.requires_grad and not self._last_requires_grad.get(n, False)]
        if newly_trainable and rank == 0:
            print(f"[EMA] {len(newly_trainable)} parameters enabled")
        for name in newly_trainable:
            param = dict(pl_module.named_parameters())[name]
            self._trainable_params.append((name, param))
            self.ema_state_dict[name] = param.detach().clone().cpu()

        newly_frozen = [n for n, p in pl_module.named_parameters()
                        if not p.requires_grad and self._last_requires_grad.get(n, False)]
        if newly_frozen and rank == 0:
            print(f"[EMA] {len(newly_frozen)} parameters frozen")
        frozen = set(newly_frozen)
        self._trainable_params = [(n, p) for n, p in self._trainable_params if n not in frozen]
        self._last_requires_grad = current

    def on_train_epoch_start(self, trainer, pl_module):
        self._refresh_trainable_params(pl_module, trainer.global_rank)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps += 1
        # Ramp the decay in so early steps are not dominated by the initial weights.
        decay = self.decay * (1 - math.exp(-self.steps / self.tau))
        with torch.no_grad():
            for name, param in self._trainable_params:
                self.ema_state_dict[name].copy_(
                    self.ema_state_dict[name] * decay + param.data.cpu() * (1.0 - decay))
            for name, buf in self._buffers:
                if buf.dtype.is_floating_point:
                    self._buffer_state[name].copy_(
                        self._buffer_state[name] * decay + buf.data.cpu() * (1.0 - decay))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        state_dict = checkpoint.get("state_dict", {})
        for name, _ in self._trainable_params:
            if name in state_dict:
                state_dict[name] = self.ema_state_dict[name].cpu().clone()
        for name, buf in self._buffers:
            if buf.dtype.is_floating_point and name in state_dict:
                state_dict[name] = self._buffer_state[name].cpu().clone()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._swap_weights(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._swap_weights(pl_module)

    def _swap_weights(self, pl_module):
        """Temporarily load the EMA weights for validation, then restore."""
        with torch.no_grad():
            for name, param in self._trainable_params:
                saved = param.data.clone()
                param.data.copy_(self.ema_state_dict[name].to(param.device))
                self.ema_state_dict[name].copy_(saved.cpu())
            for name, buf in self._buffers:
                if buf.dtype.is_floating_point:
                    saved = buf.data.clone()
                    buf.data.copy_(self._buffer_state[name].to(buf.device))
                    self._buffer_state[name].copy_(saved.cpu())


class SaveTrainConfig(pl.Callback):
    """Write the resolved configuration next to the run outputs."""

    def __init__(self, hparams):
        self.hparams = hparams

    def on_fit_start(self, trainer, pl_module):
        log_dir = Path(trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "train_config.yaml"
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.hparams, f, default_flow_style=False, allow_unicode=True)
        print(f"[config] saved to {path}")


def load_pretrained(model, path):
    """Warm-start from a checkpoint, skipping entries whose shape no longer matches."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if any(k.startswith('model.') for k in state_dict):
        state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    matched, shape_mismatch = {}, []
    for key, value in state_dict.items():
        if key not in model_dict:
            continue
        if value.shape == model_dict[key].shape:
            matched[key] = value
        else:
            shape_mismatch.append(f"{key}: checkpoint {tuple(value.shape)} "
                                  f"vs model {tuple(model_dict[key].shape)}")

    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(f"[resume] loaded {len(matched)} tensors from {path}")
    if shape_mismatch:
        print(f"[resume]   {len(shape_mismatch)} skipped (shape mismatch):")
        for line in shape_mismatch[:5]:
            print(f"    {line}")
    if missing:
        print(f"[resume]   {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[resume]   {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    init_seeds(config.SEED + 1 + local_rank, deterministic=config.DETERMINISTIC)
    print(f"[seed] base={config.SEED}, local_rank={local_rank}, "
          f"effective={config.SEED + 1 + local_rank}, deterministic={config.DETERMINISTIC}")

    world_size = len(config.DEVICES)
    grad_accum = max(math.ceil(config.NBS / (config.BATCH_SIZE * world_size)), 1)
    effective_bs = config.BATCH_SIZE * grad_accum * world_size
    print(f"[config] batch {config.BATCH_SIZE}/GPU x {world_size} GPU x "
          f"accum {grad_accum} = {effective_bs} effective "
          f"(NBS={config.NBS})")
    print(f"[config] mosaic epochs {config.MOSAIC_START}-{config.MOSAIC_END}, "
          f"MVLE epochs {config.MVLE_START_EPOCH}-{config.MVLE_END_EPOCH}")

    datamodule = VLDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        cache_images=config.CACHE_IMAGES,
    )
    model = MVLFireNet(use_mvle=True)

    resume_path = config.RESUME_CHECKPOINT
    training_resume = config.RESUME_TRAINING_CHECKPOINT
    if resume_path and training_resume:
        raise ValueError(
            "set at most one of RESUME_CHECKPOINT and RESUME_TRAINING_CHECKPOINT")

    if resume_path:
        load_pretrained(model, resume_path)
    elif training_resume:
        print(f"[resume] continuing interrupted run from {training_resume}")
    else:
        print("[resume] training from scratch")

    tb_logger = TensorBoardLogger(save_dir=config.CHECKPOINT_DIR,
                                 name=config.CHECKPOINT_NAME,
                                 version=None, log_graph=False,
                                 default_hp_metric=False)
    csv_logger = CSVLogger(save_dir=config.CHECKPOINT_DIR,
                           name=config.CHECKPOINT_NAME,
                           version=tb_logger.version)

    checkpoint_cb = ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        filename=config.CHECKPOINT_FILENAME,
        save_top_k=config.CHECKPOINT_TOP_K,
        save_last=True,
    )

    hparams = {
        'num_classes': config.NUM_CLASSES,
        'input_size': config.INPUT_SIZE,
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.NUM_WORKERS,
        'backbone_out_channels': config.BACKBONE_OUT_CHANNELS,
        'neck_out_channels': config.NECK_OUT_CHANNELS,
        'use_msa': config.USE_MSA,
        'use_cmf': config.USE_CMF,
        'decoder_hidden_dim': config.DECODER_HIDDEN_DIM,
        'decoder_num_queries': config.DECODER_NUM_QUERIES,
        'decoder_num_layers': config.DECODER_NUM_LAYERS,
        'num_denoising': config.NUM_DENOISING,
        'cls_noise_ratio': config.CLS_NOISE_RATIO,
        'box_noise_scale': config.BOX_NOISE_SCALE,
        'mvle_dim_global': config.MVLE_DIM_GLOBAL,
        'mvle_dim_local': config.MVLE_DIM_LOCAL,
        'mvle_temperature': config.MVLE_TEMPERATURE,
        'loss_weights': dict(config.LOSS_WEIGHTS),
        'use_uni_set': config.USE_UNI_SET,
        'cost_class': config.COST_CLASS,
        'cost_bbox': config.COST_BBOX,
        'cost_giou': config.COST_GIOU,
        'cost_nwd': config.COST_NWD,
        'focal_alpha': config.FOCAL_ALPHA,
        'focal_gamma': config.FOCAL_GAMMA,
        'lr': config.LR,
        'lr_mvle': config.LR_MVLE,
        'weight_decay': config.WEIGHT_DECAY,
        'momentum': config.MOMENTUM,
        'grad_clip': config.GRAD_CLIP,
        'nbs': config.NBS,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'flat_epochs': config.FLAT_EPOCHS,
        'lr_gamma': config.LR_GAMMA,
        'max_epochs': config.MAX_EPOCHS,
        'ema_decay': config.EMA_DECAY,
        'precision': config.PRECISION,
        'strategy': config.STRATEGY,
        'grad_accum': grad_accum,
        'mosaic_start': config.MOSAIC_START,
        'mosaic_end': config.MOSAIC_END,
        'mvle_start_epoch': config.MVLE_START_EPOCH,
        'mvle_end_epoch': config.MVLE_END_EPOCH,
    }

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu',
        devices=config.DEVICES,
        strategy=config.STRATEGY,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_cb,
                   ModelEMA(decay=config.EMA_DECAY),
                   MosaicControl(),
                   SaveTrainConfig(hparams)],
        gradient_clip_val=config.GRAD_CLIP,
        accumulate_grad_batches=grad_accum,
        precision=config.PRECISION,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
    )

    trainer.fit(model, datamodule=datamodule,
                ckpt_path=training_resume or None)


if __name__ == '__main__':
    main()