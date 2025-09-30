#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


def process_dataset(root: Path, save_root: Path, subsets=("On-the-go", "Robust"), force_cpu: bool = False):
    """
    Traverse <root>/<subset>/<scene>/renders/val*.png,
    split each image into left/right, save them into save_root,
    compute per-image PSNR/SSIM/LPIPS (aggregated per scene),
    and compute ONE global FID pooled across the entire dataset (all subsets).
    Uses GPU if available unless --cpu is passed.
    """
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Info] Using device: {device}")

    save_root.mkdir(parents=True, exist_ok=True)

    # Collect metrics
    per_image_records = []
    per_scene_psnr = defaultdict(list)
    per_scene_ssim = defaultdict(list)
    per_scene_lpips = defaultdict(list)

    # Global FID accumulator (all subsets pooled)
    fid_metric_overall = FrechetInceptionDistance(normalize=True).to(device)

    # Per-image metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    any_image = False

    for subset in subsets:
        subset_path = root / subset
        if not subset_path.is_dir():
            continue

        for scene_dir in subset_path.iterdir():
            renders = scene_dir / "renders"
            if not renders.is_dir():
                continue

            for img_path in renders.glob("val*.png"):
                any_image = True
                im = Image.open(img_path)
                w, h = im.size

                # Split into left (target) and right (source)
                left = im.crop((0, 0, w // 2, h))     # target (real)
                right = im.crop((w // 2, 0, w, h))    # source (fake)

                scene_name = img_path.parents[1].name
                base_name = img_path.stem
                data_id = f"{scene_name}_{base_name}"

                # Save split images
                save_left = save_root / "images"  / f"{data_id}_target.png"
                save_right = save_root / "images" / f"{data_id}_source.png"
                left.save(save_left)
                right.save(save_right)

                # Tensors: [0,1] CHW for PSNR/SSIM/FID
                t = torch.from_numpy(np.array(left.convert("RGB"))).permute(2, 0, 1).float() / 255.0
                s = torch.from_numpy(np.array(right.convert("RGB"))).permute(2, 0, 1).float() / 255.0
                t = t.to(device)
                s = s.to(device)

                # Per-image metrics
                psnr_val = psnr_metric(s, t).item()
                ssim_val = ssim_metric(s.unsqueeze(0), t.unsqueeze(0)).item()  # SSIM expects NCHW
                # LPIPS: [-1,1] NCHW
                lpips_val = lpips_metric((s * 2 - 1).unsqueeze(0), (t * 2 - 1).unsqueeze(0)).item()

                # Aggregate per scene
                scene_id = data_id.split("_val_")[0] if "_val_" in data_id else data_id
                per_scene_psnr[scene_id].append(psnr_val)
                per_scene_ssim[scene_id].append(ssim_val)
                per_scene_lpips[scene_id].append(lpips_val)

                # Record per-image row
                per_image_records.append({
                    "subset": subset,
                    "scene": scene_id,
                    "data_id": data_id,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                })

                # Global FID (pooled): targets as real, sources as fake
                fid_metric_overall.update(t.unsqueeze(0), real=True)
                fid_metric_overall.update(s.unsqueeze(0), real=False)

    # Compute global FID
    if any_image:
        try:
            fid_overall = fid_metric_overall.compute().item()
        except Exception as e:
            print(f"[Warn] Could not compute global FID: {e}")
            fid_overall = float("nan")
    else:
        fid_overall = float("nan")

    # === Save CSVs in root ===
    csv_dir = save_root
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-image metrics
    df_images = pd.DataFrame(per_image_records)
    df_images.to_csv(csv_dir / "per_image_metrics.csv", index=False)

    # 2) Per-scene metrics
    scene_rows = []
    for scene in sorted(per_scene_psnr.keys()):
        scene_rows.append({
            "scene": scene,
            "num_images": len(per_scene_psnr[scene]),
            "psnr_mean": sum(per_scene_psnr[scene]) / len(per_scene_psnr[scene]),
            "ssim_mean": sum(per_scene_ssim[scene]) / len(per_scene_ssim[scene]),
            "lpips_mean": sum(per_scene_lpips[scene]) / len(per_scene_lpips[scene]),
        })
    df_scenes = pd.DataFrame(scene_rows)
    df_scenes.to_csv(csv_dir / "per_scene_metrics.csv", index=False)

    # 3) Global FID (single row)
    df_fid_overall = pd.DataFrame([{
        "dataset": root.name,
        "fid": fid_overall,
    }])
    df_fid_overall.to_csv(csv_dir / "fid_overall.csv", index=False)

    # 4) Overall per-image means
    if not df_images.empty:
        df_overall = pd.DataFrame([{
            "psnr_mean": df_images["psnr"].mean(),
            "ssim_mean": df_images["ssim"].mean(),
            "lpips_mean": df_images["lpips"].mean(),
        }])
        df_overall.to_csv(csv_dir / "overall_means.csv", index=False)

    print(f"[Info] Metrics saved to {csv_dir}")
    print(f"[Info] Global FID: {fid_overall:.2f}" if isinstance(fid_overall, float) else "[Info] Global FID: NaN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="Root directory (SLS data)")
    parser.add_argument("--save", required=True, type=str, help="Save directory for split images")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    process_dataset(Path(args.root), Path(args.save), force_cpu=args.cpu)

