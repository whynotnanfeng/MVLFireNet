# MVLFireNet — Dataset

## FSDataset-VL

Multi-granularity text-image forest fire detection dataset introduced in the paper.
Built on top of the earlier FSDataset image collection by **adding multi-granularity textual
annotations** — no new images were introduced.

| Split | Images | Boxes | Fire | Smoke | Labels/Image |
|-------|--------|-------|------|-------|--------------|
| Train | 12,083 | 26,501 | 14,153 | 12,348 | 2.2 |
| Val | 5,640 | 10,855 | 5,986 | 4,869 | 1.9 |
| Test | 2,143 | 4,627 | 2,466 | 2,161 | 2.2 |
| **Total** | **19,866** | **41,983** | **22,605** | **19,378** | **2.1** |

### Bounding box scale distribution

| Scale | Area ratio | Proportion |
|-------|-----------|------------|
| Small | < 1% | 15.4% |
| Medium | 1–10% | 40.8% |
| Large | 10–30% | 29.9% |
| X-Large | > 30% | 13.9% |

### Multi-dimensional scene semantics

| Dimension | Categories and proportions |
|-----------|------------------------------|
| Scene | Urban (39.8%), Roadside (16.6%), Forest (13.0%), Industrial (6.6%), Grassland (6.5%), Mountain (5.2%), Indoor (3.8%), Agricultural (2.1%), Aerial (1.3%), Other (5.1%) |
| Lighting | Daytime (71.0%), Nighttime (18.6%), Dusk & Dawn (10.4%) |
| Weather | Clear (55.1%), Overcast (35.8%), Cloudy (5.5%), Foggy (2.8%), Rainy (0.8%) |
| Terrain | Flat (74.6%), Hilly (14.2%), Mountainous (6.6%), Coastal (2.4%), Riverside (1.5%), Valley (0.4%), Other (0.3%) |
| Vegetation | None (37.2%), Grass (18.3%), Deciduous (16.8%), Mixed forests (11.1%), Shrub (7.8%), Coniferous (7.4%), Crop (1.4%) |

---

## Annotation format

Two levels of text per image, which is what distinguishes FSDataset-VL from
bbox-only fire datasets:

- **Global scene description** — one per image, **80–120 words**. Describes weather, terrain,
  illumination, vegetation, and the development stage of fire/smoke. This is what the MVLE
  global pathway aligns against.
- **Local detail description** — one per bounding box, **15–30 words**. Describes the target's
  color, brightness, morphological edges, occlusion status, and diffusion. This is what the
  MVLE local pathway aligns against.

### Directory layout

```
FSDataset-VL/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── captions/
    ├── train.json
    ├── val.json
    └── test.json
```

### `captions/<split>.json`

```jsonc
{
  "images": [
    {
      "image_id": 1,
      "file_name": "fire_00001.jpg",
      "global_caption": "Overcast afternoon in a forested valley ..."
    }
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 0,
      "bbox": [0.512, 0.481, 0.143, 0.216],
      "local_caption": "Bright orange flame with well-defined edges ..."
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `image_id` | int | Joins `images` to `annotations` |
| `file_name` | str | Resolved against `images/<split>/`; only the basename is used |
| `global_caption` | str | 80–120 words, image level |
| `category_id` | int | `0 = fire`, `1 = smoke` (see `config.CLASS_NAMES`) |
| `bbox` | [4] float | **Normalized** `cx, cy, w, h` in `[0, 1]` — not `xyxy` |
| `local_caption` | str | 15–30 words, box level |

### Annotation pipeline

Text was produced by a VLM (MiMo-v2.5, Python 3.10.15) under a three-stage protocol:

1. **Generation** — global captions from image + prompt. Local captions use an
   annotation-assisted strategy: boxes are rendered onto the image (red = fire, blue = smoke)
   and the prompt requires the description to correspond to the boxed region, which avoids
   feature confusion in multi-target scenes.
2. **Automated validation** — two tiers with a feedback loop. Tier 1 runs regex/JSON parsing,
   length thresholds, and keyword-conflict checks (e.g. a caption labeled `fire` containing no
   fire-related vocabulary is rejected and regenerated). Tier 2 retries failed samples by
   feeding cropped box regions, full-image context, the draft caption and the expected category
   back to the VLM for diagnosis, then regenerates with constraint-repair prompts.
3. **Manual review** — four authors re-assessed a random 5% sample against five criteria
   (no fabricated content; color/morphology/stage alignment with visual evidence; length and
   keyword constraints; no subjective speculation; no conflict with the Fire/Smoke label).
   Corrections were applied and re-inspected, with no residual errors found.

---

## Configuration

Point the code at your copy of the dataset:

```bash
export MVLF_DATA_DIR=/path/to/FSDataset-VL
```

---

## Access

| Mirror | Link |
|--------|------|
| Quark | https://pan.quark.cn/s/ea37f9fe3bff?pwd=sf82 (code `sf82`) |
| Google Drive | https://drive.google.com/file/d/1evjqkbOR0rJnj33XlG4dDBfowXa0ZBXW/view |

Point the code at the extracted copy:

```bash
export MVLF_DATA_DIR=/path/to/FSDataset-VL
```

The images are not redistributed with this repository.
