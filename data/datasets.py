# -*- coding: utf-8 -*-
"""
Dataset and augmentation pipeline for FSDataset-VL.

Each sample carries a scene-level caption and one caption per bounding box.
"""
import os, random, cv2, numpy as np, math, json, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from tqdm import tqdm

cv2.setNumThreads(0)

# Augmentation hyperparameters, imported from config.py to keep a single source.
try:
    from config import HYP
except ImportError:
    import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import HYP

# ---------------------------------------------------------------- helpers
def letterbox(img, new_shape=640, color=(114, 114, 114), scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh + (new_shape[0] - img.shape[0] - dh)
    left, right = dw, dw + (new_shape[1] - img.shape[1] - dw)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


def stretch_resize(img, new_shape=640):
    """Resize straight to the target size, ignoring the aspect ratio."""
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
    rx, ry = new_shape[1] / w, new_shape[0] / h
    return img, (rx, ry), (0, 0)


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """HSV augmentation: additive hue shift with pure whites left untouched."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
    lut_sat[0] = 0  # keep pure white from shifting
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


# Optional Albumentations transforms. These are pixel-level only, so bounding
# boxes pass through unchanged.
_ALBUMENTATIONS_TRANSFORMS = None
_ALBUMENTATIONS_TRIED = False

def _get_albumentations():
    """Import Albumentations lazily; give up after the first failure."""
    global _ALBUMENTATIONS_TRANSFORMS, _ALBUMENTATIONS_TRIED
    if _ALBUMENTATIONS_TRIED:
        return _ALBUMENTATIONS_TRANSFORMS
    _ALBUMENTATIONS_TRIED = True
    try:
        import albumentations as A
        _ALBUMENTATIONS_TRANSFORMS = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
        ])
        # Print from the main process only, to avoid duplicate output in workers
        from torch.utils.data import get_worker_info
        if get_worker_info() is None:
            print('[FSDatasetVL] Albumentations enabled (Blur/MedianBlur/ToGray/CLAHE p=0.01)')
    except ImportError:
        from torch.utils.data import get_worker_info
        if get_worker_info() is None:
            print('[FSDatasetVL] Albumentations not installed, skipping (pip install albumentations)')
    return _ALBUMENTATIONS_TRANSFORMS


def augment_albumentations(img, labels):
    """Pixel-level Albumentations pipeline (Blur/MedianBlur/ToGray/CLAHE, p=0.01).

    No geometric transforms are used, so the boxes are returned unchanged.

    Args:
        img: uint8 RGB array of shape (H, W, 3).
        labels: array of shape (N, 5) holding [cls, x1, y1, x2, y2] in pixels.

    Returns:
        The image and the labels, the latter unmodified.
    """
    transforms = _get_albumentations()
    if transforms is None:
        return img, labels
    try:
        img = transforms(image=img)['image']
    except Exception:
        pass
    return img, labels


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
    """Random affine transform (rotate, translate, scale, shear) applied to image and boxes."""
    height, width = img.shape[0], img.shape[1]
    C = np.eye(3)
    # Move the image centre to the origin.
    C[0, 2] = -img.shape[1] / 2; C[1, 2] = -img.shape[0] / 2
    # Rotation and scale.
    R = np.eye(3)
    a = random.uniform(-degrees, degrees); s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    # Shear.
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # Move the origin back to the image centre.
    T = np.eye(3); T[0, 2] = width / 2; T[1, 2] = height / 2
    M = T @ S @ R @ C
    # Random translation.
    M[0, 2] += random.uniform(0.5 - translate, 0.5 + translate) * width
    M[1, 2] += random.uniform(0.5 - translate, 0.5 + translate) * height
    # Warp the image.
    img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,
                         borderValue=(114, 114, 114))
    # Transform the boxes: targets [N, 5] = [class, x1, y1, x2, y2]
    n = len(targets)
    if n:
        # 4 corners: (x1,y1), (x2,y1), (x2,y2), (x1,y2)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 2, 3, 4, 1, 4]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]; y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip to image bounds
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        i = box_candidates(targets[:, 1:5].T * s, xy.T, area_thr=0.01)
        targets = targets[i]; targets[:, 1:5] = xy[i]
    keep_mask = i if n else np.ones(0, dtype=bool)
    return img, targets, keep_mask


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1):
    """Drop boxes that are too small, too elongated, or changed area too much."""
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)


# ------------------------------------------------------------------ dataset
class FSDatasetVL(Dataset):
    def __init__(self, data_dir, split="train", augment=True, img_size=640, cache_images=False):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()  # accepts relative or absolute paths
        self.split = split
        self.augment = augment
        self.img_size = img_size
        self.close_mosaic = False  # toggled by a callback to disable Mosaic

        cf = self.data_dir / "captions" / f"{split}.json"
        with open(cf, "r") as f:
            self.caption_data = json.load(f)

        self.images = self.caption_data["images"]
        self.annotations = self.caption_data["annotations"]

        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.img_dir = self.data_dir / "images" / split

        # Preload every image into RAM to avoid disk I/O in the data loader.
        self.cache_images = cache_images
        self.cached_imgs = {}
        if cache_images:
            print(f'[FSDatasetVL] caching the {split} split into memory...')
            for i in tqdm(range(len(self.images)), desc=f'cache {split}'):
                img_info = self.images[i]
                fn = img_info["file_name"]
                img = cv2.imread(str(self.img_dir / Path(fn).name))
                if img is not None:
                    self.cached_imgs[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f'[FSDatasetVL] cached {len(self.cached_imgs)} images')

    def __len__(self):
        return len(self.images)

    def load_image(self, idx):
        img_info = self.images[idx]
        if self.cache_images and idx in self.cached_imgs:
            img = self.cached_imgs[idx].copy()  # copy so the cache stays pristine
        else:
            fn = img_info["file_name"]
            img = cv2.imread(str(self.img_dir / Path(fn).name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        labels = self._load_labels(img_info["image_id"], w, h)
        return img, labels, (h, w)

    def _load_labels(self, image_id, img_w, img_h):
        anns = self.img_to_anns.get(image_id, [])
        labels = []
        for a in anns:
            cx, cy, bw, bh = a["bbox"]                 # normalised cxcywh
            x1 = (cx - bw / 2) * img_w                  # to pixel xyxy
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            labels.append([a["category_id"], x1, y1, x2, y2])
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def load_mosaic(self, idx):
        """Compose four images into a2x2 canvas, then centre-crop to the target size.

        Returns:
            img4: the composed image at the target resolution.
            labels4: merged labels of shape (N, 5) holding [cls, x1, y1, x2, y2].
            indices: dataset indices of the four source images.
            valid_mask: boolean mask of the labels that survived filtering.
            per_img_keeps: for each source image, a boolean mask over its original
                annotations marking the ones random_affine kept.
            sub_regions: the rectangle each source image occupies in the crop.
        """
        s = self.img_size
        # The canvas is 2s x 2s with the crop origin jittered around its centre.
        yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        labels4 = []
        per_img_keeps = []   # per-image masks of the annotations that were kept
        canvas_regions = []  # placement of each source image on the canvas

        for i, idx_i in enumerate(indices):
            img, labels, (h, w) = self.load_image(idx_i)
            # Random affine; this also drops boxes that became degenerate.
            img, labels, keep_mask = random_affine(img, labels, degrees=HYP['degrees'],
                                                    translate=HYP['translate'], scale=HYP['scale'],
                                                    shear=HYP['shear'], border=(0, 0))
            per_img_keeps.append(keep_mask)  # record which annotations survived
            h, w = img.shape[:2]
            # Paste into the matching quadrant
            if i == 0:  # top left
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
                x2a, y2a = xc, yc
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
                x2b, y2b = w, h
            elif i == 1:  # top right
                x1a, y1a = xc, max(yc - h, 0)
                x2a, y2a = min(xc + w, s * 2), yc
                x1b, y1b = 0, h - (y2a - y1a)
                x2b, y2b = min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a = max(xc - w, 0), yc
                x2a, y2a = xc, min(s * 2, yc + h)
                x1b, y1b = w - (x2a - x1a), 0
                x2b, y2b = w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b = 0, 0
                x2b, y2b = min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            # Shift the labels into mosaic coordinates
            if len(labels):
                labels[:, 1:5:2] += x1a - x1b   # x1, x2
                labels[:, 2:5:2] += y1a - y1b   # y1, y2
                labels4.append(labels)
            # Record where this image landed on the canvas
            canvas_regions.append((x1a, y1a, x2a, y2a))

        labels4 = np.concatenate(labels4, 0) if labels4 else np.zeros((0, 5))
        # Clip to the canvas
        labels4[:, 1:5] = labels4[:, 1:5].clip(0, s * 2)

        # Centre-crop to s x s, keeping the sub-images at full resolution
        x_off = random.randint(0, s)
        y_off = random.randint(0, s)
        img4 = img4[y_off:y_off + s, x_off:x_off + s]
        labels4[:, 1:5] -= [x_off, y_off, x_off, y_off]
        labels4[:, 1:5] = labels4[:, 1:5].clip(0, s)

        # Map the canvas coordinates into the cropped image
        sub_regions = []
        for rx1, ry1, rx2, ry2 in canvas_regions:
            sub_regions.append((
                max(rx1 - x_off, 0), max(ry1 - y_off, 0),
                min(rx2 - x_off, s), min(ry2 - y_off, s),
            ))

        # Drop zero-area boxes
        valid_mask = np.ones(len(labels4), dtype=bool)
        if len(labels4):
            w_box = labels4[:, 3] - labels4[:, 1]
            h_box = labels4[:, 4] - labels4[:, 2]
            valid_mask = (w_box > 0) & (h_box > 0)
            labels4 = labels4[valid_mask]
        return img4, labels4, indices, valid_mask, per_img_keeps, sub_regions

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info["image_id"]
        gcap = img_info["global_caption"]
        is_mosaic = False

        anns = self.img_to_anns.get(image_id, [])
        lcaps = [a["local_caption"] for a in anns]

        # --- augmentation ---
        # Mosaic and CutMix are gated by close_mosaic; HSV and flip are not.
        if self.augment and not self.close_mosaic:
            # The canvas is already at the target size, so no resize is needed
            if random.random() < HYP['mosaic']:
                is_mosaic = True
                img, labels, mosaic_indices, valid_mask, per_img_keeps, _sub_regions = self.load_mosaic(idx)

                # Collect the box captions of the composited images
                # keep_mask keeps each caption paired with the box it describes
                ml_list = []
                for i_i, keep_mask in zip(mosaic_indices, per_img_keeps):
                    i_anns = self.img_to_anns.get(self.images[i_i]["image_id"], [])
                    for a, k in zip(i_anns, keep_mask):
                        if k:
                            ml_list.append(a["local_caption"])
                if len(ml_list) == len(valid_mask):
                    mosaic_local_caps = [c for c, v in zip(ml_list, valid_mask) if v]
                else:
                    mosaic_local_caps = ml_list

                # CutMix
                if random.random() < HYP['mixup']:
                    img2, labels2, mosaic_indices2, valid_mask2, per_img_keeps2, sub_regions2 = self.load_mosaic(random.randint(0, len(self) - 1))
                    r = np.random.beta(8.0, 8.0)
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    labels = np.concatenate((labels, labels2), 0)
                    # Filter the second canvas's captions the same way
                    for i_i, keep_mask in zip(mosaic_indices2, per_img_keeps2):
                        i_anns = self.img_to_anns.get(self.images[i_i]["image_id"], [])
                        for a, k in zip(i_anns, keep_mask):
                            if k:
                                mosaic_local_caps.append(a["local_caption"])
                labels = labels.reshape(-1, 5)

                # The composed captions replace those of the anchor image
                lcaps = mosaic_local_caps if mosaic_local_caps else lcaps
            else:
                img, labels, (h, w) = self.load_image(idx)
                img, ratio, pad = stretch_resize(img, self.img_size)
                if len(labels):
                    labels[:, 1] = labels[:, 1] * ratio[0] + pad[0]
                    labels[:, 2] = labels[:, 2] * ratio[1] + pad[1]
                    labels[:, 3] = labels[:, 3] * ratio[0] + pad[0]
                    labels[:, 4] = labels[:, 4] * ratio[1] + pad[1]
                labels = labels.reshape(-1, 5)
        else:
            img, labels, (h, w) = self.load_image(idx)
            img, ratio, pad = stretch_resize(img, self.img_size)
            if len(labels):
                labels[:, 1] = labels[:, 1] * ratio[0] + pad[0]
                labels[:, 2] = labels[:, 2] * ratio[1] + pad[1]
                labels[:, 3] = labels[:, 3] * ratio[0] + pad[0]
                labels[:, 4] = labels[:, 4] * ratio[1] + pad[1]

        # Albumentations, HSV and flip stay on whenever augment is set
        if self.augment:
            img, labels = augment_albumentations(img, labels)
            if random.random() < 0.5:
                img = augment_hsv(img, HYP['hsv_h'], HYP['hsv_s'], HYP['hsv_v'])
            if random.random() < HYP['flip_p']:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1], labels[:, 3] = img.shape[1] - labels[:, 3], img.shape[1] - labels[:, 1]

        # --- format conversion ---
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        # Scale to [0, 1] only; the BatchNorm layers learn the normalisation

        # xyxy -> normalised cxcywh
        if len(labels):
            x1, y1, x2, y2 = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
            cx = (x1 + x2) / 2 / self.img_size
            cy = (y1 + y2) / 2 / self.img_size
            w = (x2 - x1) / self.img_size
            h = (y2 - y1) / self.img_size
            boxes_norm = np.stack([cx, cy, w, h], axis=1)
            label_ids = labels[:, 0].astype(np.int64)
            boxes = torch.from_numpy(boxes_norm).float()
            labels_t = torch.from_numpy(label_ids).long()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.long)

        return {
            "image": img,
            "boxes": boxes,
            "labels": labels_t,
            "global_caption": gcap,
            "local_captions": lcaps,
            "image_id": image_id,
            "orig_size": torch.tensor([self.img_size, self.img_size]),
            "is_mosaic": is_mosaic,
        }


def collate_vl_fn(batch):
    images = torch.stack([i["image"] for i in batch])
    targets = [{"boxes": i["boxes"], "labels": i["labels"],
                 "orig_size": i["orig_size"]} for i in batch]
    return {
        "images": images,
        "targets": targets,
        "global_captions": [i["global_caption"] for i in batch],
        "local_captions": [i["local_captions"] for i in batch],
        "is_mosaic": [i["is_mosaic"] for i in batch],
    }


# ------------------------------------------------------------- data module
class VLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=8, cache_images=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_images = cache_images
        self._train_loader = None
        self._val_loader = None

    def setup(self, stage=None):
        self.train_dataset = FSDatasetVL(self.data_dir, 'train', augment=True, cache_images=self.cache_images)
        self.val_dataset = FSDatasetVL(self.data_dir, 'val', augment=False, cache_images=self.cache_images)

    def train_dataloader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=self.num_workers,
                                            collate_fn=collate_vl_fn, pin_memory=True,
                                            persistent_workers=self.num_workers > 0)
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                          shuffle=False, num_workers=self.num_workers,
                                          collate_fn=collate_vl_fn, pin_memory=True,
                                          persistent_workers=self.num_workers > 0)
        return self._val_loader
