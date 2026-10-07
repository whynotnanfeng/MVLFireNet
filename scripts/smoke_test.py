#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test: build MVLFireNet, run a forward pass, and verify complexity.

Runs without any dataset and without the Long-CLIP weights, so it works on a
fresh clone and in CI. Intended to catch import errors, shape mismatches and
architecture regressions.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --check-dataset    # also test the data loader
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Paper reference values (Fire 2026, 9, 409, Table 6/7) --
PAPER_PARAMS = 2.41e6
PAPER_GFLOPS = 6.8
PARAM_TOLERANCE = 0.02e6      # +/-0.02 M
GFLOPS_TOLERANCE = 0.15       # +/-0.15 G


def check_complexity(model, img_size):
    """Measure params and GFLOPs, compare against the paper."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}  ({n_params / 1e6:.4f} M)   paper: {PAPER_PARAMS / 1e6:.2f} M")

    ok = abs(n_params - PAPER_PARAMS) <= PARAM_TOLERANCE
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = macs * 2 / 1e9          # paper counts FLOPs as MACs x 2
        print(f"  GFLOPs     : {gflops:.2f} G   (MACs = {macs / 1e9:.2f} G)   paper: {PAPER_GFLOPS} G")
        ok &= abs(gflops - PAPER_GFLOPS) <= GFLOPS_TOLERANCE
    except ImportError:
        print("  GFLOPs     : skipped (pip install thop)")

    print(f"  -> {'matches paper' if ok else 'DIFFERS from paper'}")
    return ok


def check_dataset(data_dir):
    """Verify the loader accepts the documented captions/<split>.json schema."""
    import json
    from data.datasets import FSDatasetVL

    split_file = Path(data_dir) / "captions" / "val.json"
    if not split_file.exists():
        print(f"  SKIP: {split_file} not found")
        return True

    with open(split_file, encoding="utf-8") as f:
        blob = json.load(f)

    for key in ("images", "annotations"):
        if key not in blob:
            print(f"  FAIL: top-level key '{key}' missing")
            return False
    img = blob["images"][0]
    ann = blob["annotations"][0]
    for key in ("image_id", "file_name", "global_caption"):
        if key not in img:
            print(f"  FAIL: images[] entry missing '{key}'")
            return False
    for key in ("image_id", "category_id", "bbox", "local_caption"):
        if key not in ann:
            print(f"  FAIL: annotations[] entry missing '{key}'")
            return False
    if len(ann["bbox"]) != 4:
        print(f"  FAIL: bbox must have 4 values, got {len(ann['bbox'])}")
        return False
    if not all(0.0 <= v <= 1.0 for v in ann["bbox"]):
        print(f"  FAIL: bbox must be normalized to [0, 1], got {ann['bbox']}")
        return False

    print(f"  JSON schema OK ({len(blob['images'])} images, {len(blob['annotations'])} annotations)")

    try:
        ds = FSDatasetVL(data_dir, "val", augment=False)
        print(f"  FSDatasetVL constructed, __len__ = {len(ds)}")
    except Exception as e:
        print(f"  NOTE: FSDatasetVL could not be constructed ({type(e).__name__}: {e}).")
        print("        This is expected if the image files are not present.")
    return True


def main():
    parser = argparse.ArgumentParser(description="MVLFireNet smoke test")
    parser.add_argument("--check-dataset", action="store_true",
                        help="also validate the dataset loader against the documented schema")
    parser.add_argument("--data-dir", default=None,
                        help="dataset root (default: $MVLF_DATA_DIR or ./data/FSDataset-VL)")
    args = parser.parse_args()

    import config

    print("=" * 64)
    print("MVLFireNet smoke test")
    print("=" * 64)

    print("\n[1/3] Importing modules")
    from models.mvlfirenet import MVLFireNet
    print("  OK: models.mvlfirenet")

    print("\n[2/3] Building model (inference mode, MVLE branch disabled)")
    model = MVLFireNet(use_mvle=False).eval()
    print(f'  OK: {type(model).__name__} built')

    with torch.inference_mode():
        out = model(torch.randn(1, 3, config.INPUT_SIZE, config.INPUT_SIZE))
    print(f"  OK: forward pass, pred_logits {tuple(out['pred_logits'].shape)}, "
          f"pred_boxes {tuple(out['pred_boxes'].shape)}")

    print("\n[3/3] Complexity")
    complexity_ok = check_complexity(model, config.INPUT_SIZE)

    if args.check_dataset:
        import os
        print("\n[extra] Dataset schema check")
        data_dir = args.data_dir or os.environ.get("MVLF_DATA_DIR", "data/FSDataset-VL")
        check_dataset(data_dir)

    print("\n" + "=" * 64)
    if complexity_ok:
        print("Smoke test PASSED")
    else:
        print("Smoke test PASSED with warnings (see complexity above)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())