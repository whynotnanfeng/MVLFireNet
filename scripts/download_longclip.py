#!/usr/bin/env python3
"""
Download the frozen Long-CLIP checkpoint used by the MVLE branch.

Source: zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14 on HuggingFace
Architecture: ViT-L/14 (304M vision + 124M text = 428M parameters), 248-token context
Format: standard HuggingFace Transformers, loadable with CLIPModel.from_pretrained

Only the text tower is used during training, but the whole repository is
downloaded because the text weights are stored in the full checkpoint layout.

Usage:
    python scripts/download_longclip.py                    # -> weights/LongCLIP-KO-LITE
    python scripts/download_longclip.py --mirror           # use a China mirror
    python scripts/download_longclip.py -o /path/to/save   # custom output directory
    python scripts/download_longclip.py --skip-verify      # do not verify the load

Requires: pip install huggingface_hub transformers
"""
import argparse
import os
import sys

REPO_ID = "zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14"

# Formats this project does not load. The .pt files are pickle-based, so they are
# excluded deliberately rather than merely for being unused.
IGNORE_PATTERNS = [
    "*.pt",
    "Long-ViT-L-14-KO-LITE-*",
    ".gitattributes",
]


def main():
    parser = argparse.ArgumentParser(description="Download the Long-CLIP checkpoint")
    parser.add_argument("--output", "-o", default="weights/LongCLIP-KO-LITE",
                        help="output directory (default: weights/LongCLIP-KO-LITE)")
    parser.add_argument("--mirror", "-m", action="store_true",
                        help="use the hf-mirror.com endpoint")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the load verification step")
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[info] using the hf-mirror.com endpoint")

    save_dir = os.path.abspath(args.output)
    print(f"[info] repository: {REPO_ID}")
    print(f"[info] destination: {save_dir}")
    print(f"[info] skipping: {IGNORE_PATTERNS}\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] huggingface_hub is missing: pip install huggingface_hub")
        sys.exit(1)

    print("[1/2] downloading...")
    path = snapshot_download(REPO_ID, local_dir=save_dir,
                             ignore_patterns=IGNORE_PATTERNS)
    print(f"[1/2] done: {path}\n")

    print("[info] downloaded files:")
    for name in sorted(os.listdir(save_dir)):
        if name.startswith('.'):
            continue
        size_mb = os.path.getsize(os.path.join(save_dir, name)) / 1024 / 1024
        print(f"  {name:45s} {size_mb:>8.1f} MB")
    print()

    if args.skip_verify:
        print("[2/2] verification skipped (--skip-verify)")
        return

    print("[2/2] verifying the load...")
    try:
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(save_dir)
    except Exception as exc:
        print(f"[error] verification failed: {exc}")
        sys.exit(1)

    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"  vision encoder: {vision_params / 1e6:.1f}M "
          f"(ViT-L/14, hidden={model.vision_model.config.hidden_size}, "
          f"layers={model.vision_model.config.num_hidden_layers})")
    print(f"  text encoder:   {text_params / 1e6:.1f}M "
          f"(hidden={model.text_model.config.hidden_size}, "
          f"layers={model.text_model.config.num_hidden_layers}, "
          f"max_pos={model.text_model.config.max_position_embeddings})")
    print(f"  total:          {total / 1e6:.1f}M")
    print("[2/2] verification passed\n")
    print("Point the training run at it with:")
    print(f"    export MVLF_CLIP_PATH={save_dir}")


if __name__ == "__main__":
    main()