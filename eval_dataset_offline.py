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


KNOWN_SUBSETS = {"on-the-go", "robust"}  # lowercase matching


def infer_scene_id(data_id: str) -> str:
    return data_id.split("_val_")[0] if "_val_" in data_id else data_id


def detect_subset(path: Path) -> str:
    # Look at parent folders for subset names
    for parent in path.parents:
        if parent.name.lower() in KNOWN_SUBSETS:
            return parent.name  # keep original casing
    return "unknown"


def compute_metrics_from_saved(save_dir: Path, force_cpu: bool = False):
    """
    Read *_target.png and *_source.png pairs under save_dir,
    compute PSNR/SSIM/LPIPS per image, aggregate per scene,
    and compute global FID. Results are saved as CSVs in save_dir.
    """
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Info] Using device: {device}")

    # Collect target-source pairs
    target_paths = sorted((save_dir / "images").rglob("*_target.png"))
    pairs = []
    for tpath in target_paths:
        stem = tpath.stem[:-7]  # remove "_target"
        spath = tpath.with_name(f"{stem}_source.png")
        if spath.exists():
            pairs.append((stem, tpath, spath))
        else:
            print(f"[Warn] Missing source for {tpath.name}; skipping")

    if not pairs:
        print("[Error] No valid (target, source) pairs found in save_dir.")
        return

    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    # Aggregators
    per_image_records = []
    per_scene_psnr = defaultdict(list)
    per_scene_ssim = defaultdict(list)
    per_scene_lpips = defaultdict(list)

    # Loop through all pairs
    for data_id, tpath, spath in pairs:
        subset = detect_subset(tpath)
        scene_id = infer_scene_id(data_id)

        # Load images
        t_img = Image.open(tpath).convert("RGB")
        s_img = Image.open(spath).convert("RGB")

        # Convert to tensors [0,1]
        t = torch.from_numpy(np.array(t_img)).permute(2, 0, 1).float() / 255.0
        s = torch.from_numpy(np.array(s_img)).permute(2, 0, 1).float() / 255.0
        t = t.to(device)
        s = s.to(device)

        # Per-image metrics
        psnr_val = psnr_metric(s, t).item()
        ssim_val = ssim_metric(s.unsqueeze(0), t.unsqueeze(0)).item()
        lpips_val = lpips_metric((s * 2 - 1).unsqueeze(0), (t * 2 - 1).unsqueeze(0)).item()

        # Aggregate
        per_scene_psnr[scene_id].append(psnr_val)
        per_scene_ssim[scene_id].append(ssim_val)
        per_scene_lpips[scene_id].append(lpips_val)

        per_image_records.append({
            "subset": subset,
            "scene": scene_id,
            "data_id": data_id,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "target_path": str(tpath),
            "source_path": str(spath),
        })

        # Update FID
        fid_metric.update(t.unsqueeze(0), real=True)
        fid_metric.update(s.unsqueeze(0), real=False)

    # Compute global FID
    try:
        fid_value = fid_metric.compute().item()
    except Exception as e:
        print(f"[Warn] Could not compute FID: {e}")
        fid_value = float("nan")

    # Save CSVs
    df_images = pd.DataFrame(per_image_records)
    df_images.to_csv(save_dir / "per_image_metrics.csv", index=False)

    scene_rows = []
    for scene in sorted(per_scene_psnr.keys()):
        rows = [r for r in per_image_records if r["scene"] == scene]
        subset = rows[0]["subset"] if rows else "unknown"
        scene_rows.append({
            "subset": subset,
            "scene": scene,
            "num_images": len(per_scene_psnr[scene]),
            "psnr_mean": sum(per_scene_psnr[scene]) / len(per_scene_psnr[scene]),
            "ssim_mean": sum(per_scene_ssim[scene]) / len(per_scene_ssim[scene]),
            "lpips_mean": sum(per_scene_lpips[scene]) / len(per_scene_lpips[scene]),
        })
    pd.DataFrame(scene_rows).to_csv(save_dir / "per_scene_metrics.csv", index=False)

    pd.DataFrame([{"fid": fid_value}]).to_csv(save_dir / "fid_overall.csv", index=False)

    pd.DataFrame([{
        "psnr_mean": df_images["psnr"].mean(),
        "ssim_mean": df_images["ssim"].mean(),
        "lpips_mean": df_images["lpips"].mean(),
    }]).to_csv(save_dir / "overall_means.csv", index=False)

    print(f"[Info] Saved CSVs to {save_dir}")
    print(f"[Info] Global FID: {fid_value:.2f}" if isinstance(fid_value, float) else "[Info] Global FID: NaN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", required=True, type=str, help="Directory containing *_target.png and *_source.png")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    compute_metrics_from_saved(Path(args.save), force_cpu=args.cpu)

