#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


def eval_from_splits(save_root: Path, force_cpu: bool = False):
    """
    Evaluate metrics from already-split images in save_root/'images':
      <scene>_<basename>_target.png (real) and
      <scene>_<basename>_source.png (fake).
    Saves CSVs into save_root.
    """
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Info] [EVAL] Using device: {device}")

    images_dir = save_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect pairs
    target_paths = sorted(images_dir.rglob("*_target.png"))
    pairs = []
    for tpath in target_paths:
        stem = tpath.stem[:-7]  # remove "_target"
        spath = tpath.with_name(f"{stem}_source.png")
        if spath.exists():
            pairs.append((stem, tpath, spath))
        else:
            print(f"[Warn] Missing source for {tpath.name}; skipping")

    if not pairs:
        print("[Error] No valid (target, source) pairs found in save_root/images.")
        return

    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    fid_metric_overall = FrechetInceptionDistance(normalize=True).to(device)

    # Aggregators
    per_image_records = []
    per_scene_psnr = defaultdict(list)
    per_scene_ssim = defaultdict(list)
    per_scene_lpips = defaultdict(list)

    # Helpers
    def infer_scene_id(data_id: str) -> str:
        return data_id.split("_val_")[0] if "_val_" in data_id else data_id

    # Loop
    for data_id, tpath, spath in pairs:
        t_img = Image.open(tpath).convert("RGB")
        s_img = Image.open(spath).convert("RGB")

        t = torch.from_numpy(np.array(t_img)).permute(2, 0, 1).float() / 255.0
        s = torch.from_numpy(np.array(s_img)).permute(2, 0, 1).float() / 255.0
        t = t.to(device)
        s = s.to(device)

        psnr_val = psnr_metric(s, t).item()
        ssim_val = ssim_metric(s.unsqueeze(0), t.unsqueeze(0)).item()
        lpips_val = lpips_metric((s * 2 - 1).unsqueeze(0), (t * 2 - 1).unsqueeze(0)).item()

        scene_id = infer_scene_id(data_id)
        per_scene_psnr[scene_id].append(psnr_val)
        per_scene_ssim[scene_id].append(ssim_val)
        per_scene_lpips[scene_id].append(lpips_val)

        per_image_records.append({
            "scene": scene_id,
            "data_id": data_id,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "target_path": str(tpath),
            "source_path": str(spath),
        })

        fid_metric_overall.update(t.unsqueeze(0), real=True)
        fid_metric_overall.update(s.unsqueeze(0), real=False)

    try:
        fid_overall = fid_metric_overall.compute().item()
    except Exception as e:
        print(f"[Warn] Could not compute global FID: {e}")
        fid_overall = float("nan")

    # Save CSVs to save_root
    df_images = pd.DataFrame(per_image_records)
    df_images.to_csv(save_root / "per_image_metrics.csv", index=False)

    scene_rows = []
    for scene in sorted(per_scene_psnr.keys()):
        scene_rows.append({
            "scene": scene,
            "num_images": len(per_scene_psnr[scene]),
            "psnr_mean": sum(per_scene_psnr[scene]) / len(per_scene_psnr[scene]),
            "ssim_mean": sum(per_scene_ssim[scene]) / len(per_scene_ssim[scene]),
            "lpips_mean": sum(per_scene_lpips[scene]) / len(per_scene_lpips[scene]),
        })
    pd.DataFrame(scene_rows).to_csv(save_root / "per_scene_metrics.csv", index=False)

    pd.DataFrame([{
        "dataset": save_root.name,
        "fid": fid_overall,
    }]).to_csv(save_root / "fid_overall.csv", index=False)

    if not df_images.empty:
        pd.DataFrame([{
            "psnr_mean": df_images["psnr"].mean(),
            "ssim_mean": df_images["ssim"].mean(),
            "lpips_mean": df_images["lpips"].mean(),
        }]).to_csv(save_root / "overall_means.csv", index=False)

    print(f"[Info] [EVAL] Metrics saved to {save_root}")
    print(f"[Info] [EVAL] Global FID: {fid_overall:.2f}" if isinstance(fid_overall, float) else "[Info] [EVAL] Global FID: NaN")


def split_dataset(root: Path, save_root: Path, subsets=("On-the-go", "Robust")):
    """
    Traverse NEW layout:
        root/<subset>/<scene>/<scene>-All/SLS/renders/val*.png
    and split each image into left/right → save_root/'images'.
    """
    out_dir = save_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_saved = 0
    for subset in subsets:
        subset_path = root / subset
        if not subset_path.is_dir():
            continue

        for scene_dir in subset_path.iterdir():
            if not scene_dir.is_dir():
                continue

            scene_name = scene_dir.name  # variable part
            # NEW exact path using the scene variable:
            renders = scene_dir / f"{scene_name}-All" / "SLS" / "renders"
            if not renders.is_dir():
                continue

            for img_path in renders.glob("val*.png"):
                im = Image.open(img_path)
                w, h = im.size

                # Split: left=target (real), right=source (fake)
                left = im.crop((0, 0, w // 2, h))
                right = im.crop((w // 2, 0, w, h))

                base_name = img_path.stem
                data_id = f"{scene_name}_{base_name}"

                save_left = out_dir / f"{data_id}_target.png"
                save_right = out_dir / f"{data_id}_source.png"
                left.save(save_left)
                right.save(save_right)
                num_saved += 2

    print(f"[Info] Split images saved to {out_dir} (files written: {num_saved})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="Root directory (SLS data)")
    parser.add_argument("--save", required=True, type=str, help="Save directory for split images / metrics")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--eval", action="store_true",
                        help="If set: split + eval; If not set: only split (no metrics)")
    args = parser.parse_args()

    root = Path(args.root)
    save_root = Path(args.save)

    # Always split from the NEW structure
    split_dataset(root, save_root, ["000", "001", "002"])

    # If --eval, evaluate from the splits we just wrote
    if args.eval:
        eval_from_splits(save_root, force_cpu=args.cpu)


if __name__ == "__main__":
    main()

