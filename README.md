# MVLFireNet

[![Paper](https://img.shields.io/badge/paper-Fire%202026%2C%209%2C%20409-8B0000)](https://www.mdpi.com/2571-6255/9/9/409)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-FSDataset--VL-blue)](docs/DATASET.md)

Lightweight forest fire and smoke detection for UAV-based monitoring, with
multi-granularity vision-language enhancement.

**2.41 M parameters . 6.8 GFLOPs . 86.3% mAP@0.5 . 54.8% mAP@0.5:0.95**

Paper: [A Lightweight Forest Fire Detection Model with Multi-Granularity
Vision-Language Enhancement](https://www.mdpi.com/2571-6255/9/9/409) --
*Fire* **2026**, 9, 409 . DOI [10.3390/fire9090409](https://doi.org/10.3390/fire9090409)

---

## What the paper adds

Forest fire targets are small and highly variable in scale, sit in cluttered
backgrounds, and must run on power-limited UAV hardware. Three problems get in the
way, and each module below addresses one:

| Module | Paper | Problem it solves | Code |
|--------|-------|-------------------|------|
| **MSA** | Section 2.2.2 | Self-attention flattens 2D feature maps into 1D sequences, destroying the high-frequency spatial detail that weak fire spots and smoke edges depend on. | [`models/modules.py`](models/modules.py) -> `MSAAttention` |
| **CMF** | Section 2.2.3 | Plain concatenation at pyramid nodes treats shallow and deep features as interchangeable, although they live in different representation spaces. | [`models/modules.py`](models/modules.py) -> `CMF` |
| **MVLE** | Section 2.2.4 | Pixel-only detectors confuse fire with reddish leaves, sunset glow, morning fog and dust. | [`models/mvle.py`](models/mvle.py) -> `MVLEBranch` |

The backbone, neck and head follow Section 2.2.1: an ELAN backbone with SPPF
(`Backbone`, `SPPF`), an FPN neck (`FPNNeck`), and an RT-DETR decoder
(`RTDETRDecoder`). Each class docstring cites the section and equations it
implements.

**MVLE is discarded at inference.** It contributes gradients during training and
nothing to the deployed model, so the reported 2.41 M / 6.8 G figures cover the
whole detector.

---

## Architecture

```
Input 640x640
  |
  +-- Backbone    ELAN + SPPF                        -> C3, C4, C5   (1.40 M)
  |
  +-- Neck        FPN
  |     P5: C5 -> MSABlock (MSA)                     -> P5
  |     P4: CMF(C4, P5 up) -> ELANBlock               -> P4         (0.43 M total)
  |     P3: CMF(C3, P4 up) -> ELANBlock               -> P3
  |
  +-- Head        RT-DETR decoder, 300 object queries                (0.58 M)
  |               MAL classification + L1 + GIoU + denoising training
  |
  +-- MVLE        dual-pathway alignment, training only, dropped at inference
        global: C5 -> Conv1x1(128) -> PE -> MHSA -> gated pool -> 128-d
        local:  C5 -> Conv1x1(64)  -> PE -> MHSA -> gated pool -> 64-d
        text:   frozen Long-CLIP -> shared trunk -> {global head, local head}
        loss:   symmetric InfoNCE, tau = 0.1
```

### Complexity check

Measured with `thop` at 640x640, GFLOPs = MACs x 2:

| Configuration | Params | FLOPs |
|---------------|--------|-------|
| Baseline (no MSA, no CMF) | 2.4356 M | 6.63 G |
| + MSA | 2.3768 M | 6.58 G |
| + CMF | 2.4715 M | 6.87 G |
| **Full (MSA + CMF)** | **2.4127 M** | **6.82 G** |

Reproduce with `python scripts/smoke_test.py`.

---

## Installation

Python >= 3.10, PyTorch >= 2.4. The paper used PyTorch 2.4.1 + cu118 on an RTX 3090.

```bash
git clone https://github.com/whynotnanfeng/MVLFireNet.git
cd MVLFireNet

python -m venv .venv && source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
torch>=2.4.0
torchvision>=0.19.0
lightning>=2.4.0
transformers>=4.40.0
opencv-python>=4.9.0
numpy>=1.26.0
tqdm>=4.66.0
albumentations>=1.4.0   # optional, skipped if absent
thop>=0.1.1             # optional, FLOPs counting
huggingface_hub>=0.23.0 # download_longclip.py
```
</details>

---

## Dataset

**FSDataset-VL** -- 19,866 images / 41,983 boxes, split 6:3:1 into
train 12,083 / val 5,640 / test 2,143. Two classes: `fire` and `smoke`.

Each image carries a scene-level caption (80-120 words); each box carries a
target-level caption (15-30 words). The MVLE branch aligns visual features with
both.

| Mirror | Link |
|--------|------|
| Quark | https://pan.quark.cn/s/ea37f9fe3bff?pwd=sf82 (code `sf82`) |
| Google Drive | https://drive.google.com/file/d/1evjqkbOR0rJnj33XlG4dDBfowXa0ZBXW/view |

Extract into this layout:

```
FSDataset-VL/
+---- images/
|   +---- train/
|   +---- val/
|   `---- test/
`---- captions/
    +---- train.json
    +---- val.json
    `---- test.json
```

`captions/<split>.json`:

```jsonc
{
  "images": [
    { "image_id": 1, "file_name": "xxx.jpg", "global_caption": "80-120 words ..." }
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 0,            // 0 = fire, 1 = smoke
      "bbox": [0.51, 0.48, 0.14, 0.22],   // normalised cx, cy, w, h
      "local_caption": "15-30 words ..."
    }
  ]
}
```

Full field reference in [`docs/DATASET.md`](docs/DATASET.md).

---

## Configuration

Paths are set through environment variables, so the source does not need editing:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MVLF_DATA_DIR` | Dataset root | `./data/FSDataset-VL` |
| `MVLF_PROJECT_DIR` | Project root | repository root |
| `MVLF_CLIP_PATH` | Frozen Long-CLIP checkpoint | `./weights/LongCLIP-KO-LITE` |
| `MVLF_CHECKPOINT_DIR` | Runs and checkpoints | `./runs` |

```bash
export MVLF_DATA_DIR=/path/to/FSDataset-VL            # Linux / macOS
$env:MVLF_DATA_DIR="D:\data\FSDataset-VL"             # PowerShell
```

Everything else lives in [`config.py`](config.py).

### Long-CLIP weights

MVLE needs a Long-CLIP checkpoint (248-token context). It is downloaded on demand,
never committed:

```bash
python scripts/download_longclip.py                 # -> weights/LongCLIP-KO-LITE
python scripts/download_longclip.py --mirror        # China mirror
python scripts/download_longclip.py -o /custom/path
```

### Key hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `MAX_EPOCHS` | 300 | Training epochs |
| `BATCH_SIZE` | 32 | Per-GPU batch size |
| `NBS` | 256 | Nominal batch size; gradient accumulation derives from it |
| `LR` / `LR_MVLE` | 1e-3 | Base learning rate, cosine-annealed after warmup and a flat phase |
| `DEVICES` / `STRATEGY` | `[0]` / `auto` | Set `DEVICES=[0, 1]` and `STRATEGY="ddp"` for multi-GPU |
| `INPUT_SIZE` | 640 | Training input resolution |
| `USE_MSA` / `USE_CMF` | `True` / `True` | Toggle the two neck contributions |
| `MVLE_START_EPOCH` / `MVLE_END_EPOCH` | 0 / 999 | Epoch window for the alignment losses |
| `MOSAIC_START` / `MOSAIC_END` | 0 / 284 | Epoch window for Mosaic augmentation |
| `LOSS_WEIGHTS` | see file | GIoU weighted above L1; `config.py` explains why |

---

## Training

```bash
python train.py
```

Checkpoints and TensorBoard logs are written to `$MVLF_CHECKPOINT_DIR/mvlfirenet/`,
with the resolved configuration dumped alongside as `train_config.yaml`.

```bash
tensorboard --logdir runs
```

To fine-tune from existing weights set `RESUME_CHECKPOINT`; to continue an
interrupted run set `RESUME_TRAINING_CHECKPOINT`. Set at most one of the two.

```bash
# verify the environment without a dataset or the CLIP weights
python scripts/smoke_test.py
```

### Reproducing the ablation study

Set `USE_MSA` and `USE_CMF` in `config.py`, and zero the alignment losses in
`LOSS_WEIGHTS` for rows that exclude MVLE:

| Row | `USE_MSA` | `USE_CMF` | `loss_clip_global` / `loss_clip_local` |
|-----|-----------|-----------|---------------------------------------|
| Baseline | `False` | `False` | `0.0` |
| + MSA | `True` | `False` | `0.0` |
| + CMF | `False` | `True` | `0.0` |
| + MVLE | `False` | `False` | keep `2.0` |
| Full | `True` | `True` | keep `2.0` |

`+ MVLE` has the same complexity as `Baseline`: that row is what demonstrates the
zero inference overhead of the branch.

---

## Repository layout

```
MVLFireNet/
|-- CITATION.cff           # Citation File Format (GitHub "Cite" button)
|-- CITATION.bib           # BibTeX entry for the paper
|-- config.py              # paths, data, hardware, architecture, loss, optimisation
|-- train.py               # training entry: EMA, Mosaic schedule, grad accumulation
|-- loss.py                # MAL + Hungarian / union matching + L1 + GIoU + NWD
|-- metrics.py             # mAP, per-image TP/FP, confusion matrix
|-- utils.py               # box coordinate helpers
|-- models/
|   |-- mvlfirenet.py      # MVLFireNet: backbone, wiring, LightningModule, optimiser
|   |-- modules.py         # MSAAttention, MSABlock, CMF, ELANBlock, SPPF, MGFFN
|   |-- neck.py            # FPN neck with MSA at P5 and CMF at the fusion nodes
|   |-- head.py            # RT-DETR decoder: 300 queries, deformable attention, denoising
|   `-- mvle.py            # MVLE: dual-pathway alignment, frozen Long-CLIP text encoder
|-- data/
|   `-- datasets.py        # FSDatasetVL dataset, Mosaic/HSV/flip/albumentations
|-- scripts/
|   |-- download_longclip.py
|   `-- smoke_test.py      # complexity check against the paper
`-- docs/
    `-- DATASET.md         # annotation schema and statistics
```

---

## Citation

Cite the paper as MDPI formats it (the same string the journal page shows):

```
Ma, Y.; Shan, W.; Sui, Y.; Wang, M. A Lightweight Forest Fire Detection Model with Multi-Granularity Vision-Language Enhancement. Fire 2026, 9, 409. https://doi.org/10.3390/fire9090409
```

Citation-manager formats are provided as well, so you can import the entry
directly instead of typing it:

* [`CITATION.cff`](CITATION.cff) -- Citation File Format 1.2.0. GitHub reads
  this and renders a "Cite" button on the repository.
* [`CITATION.bib`](CITATION.bib) -- BibTeX, for Zotero, JabRef and BibDesk.

If you would rather take the citation straight from the publisher, the
journal page is <https://www.mdpi.com/2571-6255/9/9/409>.

```bibtex
@article{ma2026mvlfirenet,
  author  = {Ma, Yifan and Shan, Weifeng and Sui, Yanwei and Wang, Mengyu},
  title   = {A Lightweight Forest Fire Detection Model with Multi-Granularity Vision-Language Enhancement},
  journal = {Fire},
  year    = {2026},
  volume  = {9},
  number  = {9},
  pages   = {409},
  doi     = {10.3390/fire9090409},
  issn    = {2571-6255},
  publisher = {MDPI AG},
}
```

## License

Code is released under the [MIT License](LICENSE).

The article is (c) 2026 by the authors under CC BY 4.0. The Long-CLIP weights
downloaded by `scripts/download_longclip.py` carry their own license and are not
redistributed here. FSDataset-VL images are not included in this repository; see
[`docs/DATASET.md`](docs/DATASET.md) for access details.