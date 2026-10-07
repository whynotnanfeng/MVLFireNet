# -*- coding: utf-8 -*-
"""
Configuration for MVLFireNet.

Organised as: paths -> data -> hardware -> backbone -> neck -> head -> MVLE
-> loss -> matching -> optimisation -> logging.

Every path can be overridden through an environment variable, so this file does
not normally need to be edited:

===========================  ==================================================
``MVLF_DATA_DIR``           dataset root
``MVLF_PROJECT_DIR``        project root
``MVLF_CLIP_PATH``          frozen Long-CLIP checkpoint
``MVLF_CHECKPOINT_DIR``     where runs and checkpoints are written
===========================  ==================================================
"""
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_DIR = Path(os.environ.get("MVLF_PROJECT_DIR", Path(__file__).resolve().parent))
DATA_DIR = os.environ.get("MVLF_DATA_DIR", str(PROJECT_DIR / "data/FSDataset-VL"))
CLIP_MODEL_PATH = os.environ.get("MVLF_CLIP_PATH", str(PROJECT_DIR / "weights/LongCLIP-KO-LITE"))
CHECKPOINT_DIR = os.environ.get("MVLF_CHECKPOINT_DIR", str(PROJECT_DIR / "runs"))

# Reduce batch size and feature widths for debugging on a small GPU.
DEBUG = False

# -- Dataset -----------------------------------------------------------
NUM_CLASSES = 2                    # 0 = fire, 1 = smoke
CLASS_NAMES = ["fire", "smoke"]
INPUT_SIZE = 640
BATCH_SIZE = 32                    # per GPU
NUM_WORKERS = 4
CACHE_IMAGES = True                # preload images into RAM

HYP = {
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
    'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0,
    'flip_p': 0.5, 'mosaic': 0.35, 'mixup': 0.0,
}

# -- Hardware ----------------------------------------------------------
DEVICES = [0]                      # e.g. [0, 1] for two GPUs
STRATEGY = "auto"                  # "ddp" when using multiple GPUs
PRECISION = "16-mixed"
SEED = 42
DETERMINISTIC = True

# -- Backbone ----------------------------------------------------------
# Channels of the C3, C4 and C5 outputs (strides 8, 16 and 32).
BACKBONE_OUT_CHANNELS = [128, 128, 256]

# -- Neck --------------------------------------------------------------
NECK_OUT_CHANNELS = [64, 128, 256]     # P3, P4, P5

# Paper contributions. Both flags exist so the baseline configuration of the
# ablation study can be expressed without editing code elsewhere; the released
# model has both enabled.
USE_MSA = True      # Multi-Scale Spatial-Aware attention at P5
USE_CMF = True      # Cross-Modulation Fusion at the P4 and P3 nodes

# -- Head --------------------------------------------------------------
DECODER_HIDDEN_DIM = 128
DECODER_NHEAD = 4
DECODER_NUM_QUERIES = 300
DECODER_NUM_LAYERS = 2

# Denoising training: the decoder also sees ground truth perturbed by known
# label and box noise alongside the matched predictions.
NUM_DENOISING = 100
CLS_NOISE_RATIO = 0.5
BOX_NOISE_SCALE = 1.0

# -- MVLE branch (training only) ----------------------------------------
MVLE_DIM_GLOBAL = 128
MVLE_DIM_LOCAL = 64
MVLE_USE_POS_EMBED = True
MVLE_TEMPERATURE = 0.1
MVLE_TEMPERATURE_LOCAL = 0.1

# Hidden size of the frozen Long-CLIP text encoder.
TEXT_ENCODER_DIM = 768

# Epoch window over which the alignment losses are active.
MVLE_START_EPOCH = 0
MVLE_END_EPOCH = 999                # 999 keeps them active for the whole run
MVLE_WARMUP_EPOCHS = 3              # ramp the loss in linearly

# Mosaic augmentation window.
MOSAIC_START = 0
MOSAIC_END = 284

# -- Loss weights ------------------------------------------------------
# Fire and smoke boundaries are amorphous and annotated subjectively, so
# overlap quality matters more than exact coordinates: GIoU outweighs L1.
# NWD is used for matching only (see COST_NWD) and has no loss term.
LOSS_WEIGHTS = {
    "loss_ce": 2.5,
    "loss_bbox": 2.0,
    "loss_giou": 5.0,
    "loss_nwd": 0.0,
    "loss_clip_global": 2.0,
    "loss_clip_local": 2.0,
    "loss_ce_dn": 2.5,
    "loss_bbox_dn": 2.0,
    "loss_giou_dn": 5.0,
    "loss_nwd_dn": 0.0,
}

# -- Hungarian matching ------------------------------------------------
COST_CLASS = 2.5
COST_BBOX = 2.0
COST_GIOU = 5.0
COST_NWD = 2.0                     # assists pairing of small boxes
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 1.5
EOS_COEF = 0.1
USE_UNI_SET = True                 # share box losses across decoder layers

# -- Optimisation ------------------------------------------------------
MAX_EPOCHS = 300
WARMUP_EPOCHS = 3
FLAT_EPOCHS = 30                   # hold the base LR before annealing
LR = 1e-3
LR_MVLE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
GRAD_CLIP = 1.0
NBS = 256                          # nominal batch size, sets grad accumulation
EMA_DECAY = 0.9999
LR_GAMMA = 0.5                     # final LR is base LR times this

# -- Logging -----------------------------------------------------------
LOG_EVERY_N_STEPS = 60
CHECKPOINT_NAME = "mvlfirenet"
CHECKPOINT_MONITOR = "val_map"
CHECKPOINT_MODE = "max"
CHECKPOINT_TOP_K = 2
CHECKPOINT_FILENAME = "mvlfirenet-{epoch:02d}-{val_map:.2f}"

# Set exactly one of these to resume or fine-tune.
RESUME_CHECKPOINT = ""            # warm start from the given weights
RESUME_TRAINING_CHECKPOINT = ""   # resume an interrupted run

# -- Evaluation --------------------------------------------------------
EVAL_CONF_THRESHOLD = 0.001

if DEBUG:
    BATCH_SIZE = 2
    DECODER_NUM_QUERIES = 100
    NECK_OUT_CHANNELS = [128, 128, 128]