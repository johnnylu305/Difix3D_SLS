#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load an image file -> float tensor in [-1, 1], shape 1x3xHxW."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)  # HxWx3, uint8
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # 3xHxW in [0,1]
    t = (t * 2.0 - 1.0).unsqueeze(0).to(device)  # 1x3xHxW in [-1,1]
    return t


def build_split_entries(split_dir: Path, device: torch.device, lpips_metric) -> dict:
    """
    From <split_dir>/images, find pairs:
      *_target.png and *_source.png
    Returns: { data_id: {image, target_image, ref_image, prompt, lpips} }
    """
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        print(f"[Warn] Missing images directory: {images_dir}")
        return {}

    entries = {}
    target_paths = sorted(images_dir.glob("*_target.png"))
    for tpath in target_paths:
        stem = tpath.stem[:-7]  # drop "_target"
        spath = tpath.with_name(f"{stem}_source.png")
        if not spath.exists():
            print(f"[Warn] Missing source for target: {tpath.name} (skipping)")
            continue

        # Compute LPIPS(source, target)
        src = load_rgb_tensor(spath, device)   # fake / input
        tgt = load_rgb_tensor(tpath, device)   # real / gt
        with torch.no_grad():
            lp = lpips_metric(src, tgt).item()

        data_id = stem  # e.g., <scene>_<basename>
        entries[data_id] = {
            "image": str(spath.resolve()),          # source (input)
            "target_image": str(tpath.resolve()),   # target (gt)
            "ref_image": None,
            "prompt": "remove degradation",
            "lpips": lp,
        }

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset.json next to Train/ and Val/ with abs paths and per-pair LPIPS."
    )
    parser.add_argument("--root", required=True, type=str,
                        help="Parent folder containing Train/ and Val/ directories.")
    parser.add_argument("--train-name", default="Train", type=str,
                        help="Folder name for train split under --root (default: Train)")
    parser.add_argument("--val-name", default="Val", type=str,
                        help="Folder name for val split under --root (mapped to 'test' in JSON; default: Val)")
    parser.add_argument("--out-name", default="dataset.json", type=str,
                        help="Output JSON filename written at --root level (default: dataset.json)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    parser.add_argument("--lpips-backbone", default="alex", choices=["alex", "vgg", "squeeze"],
                        help="Backbone for LPIPS (default: alex; alex is fastest)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_dir = root / args.train_name
    val_dir = root / args.val_name
    out_path = root / args.out_name

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Info] Using device: {device}")

    # LPIPS metric
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_backbone).to(device)
    lpips_metric.eval()

    manifest = {"train": {}, "test": {}}

    # Train
    if train_dir.is_dir():
        print(f"[Info] Scanning train split in: {train_dir}")
        manifest["train"] = build_split_entries(train_dir, device, lpips_metric)
        print(f"[Info] Train entries: {len(manifest['train'])}")
    else:
        print(f"[Warn] Train dir not found: {train_dir}")

    # Val -> test
    if val_dir.is_dir():
        print(f"[Info] Scanning val split in: {val_dir}")
        manifest["test"] = build_split_entries(val_dir, device, lpips_metric)
        print(f"[Info] Test entries: {len(manifest['test'])}")
    else:
        print(f"[Warn] Val dir not found: {val_dir}")

    # Write JSON
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] Wrote manifest with LPIPS to: {out_path}")


if __name__ == "__main__":
    main()

