#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def load_manifest(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r") as f:
        manifest = json.load(f)

    rows = []
    for split in ("train", "test"):
        d = manifest.get(split, {})
        if not isinstance(d, dict):
            continue
        for data_id, entry in d.items():
            lp = entry.get("lpips", None)
            src = entry.get("image") or entry.get("source")
            tgt = entry.get("target_image") or entry.get("target")
            if lp is None or src is None or tgt is None:
                continue
            rows.append({
                "split": split,
                "data_id": data_id,
                "lpips": float(lp),
                "source": src,
                "target": tgt,
            })
    if not rows:
        raise ValueError("No usable entries found in the JSON (missing lpips/source/target).")
    return pd.DataFrame(rows)


def concat_side_by_side(src_path: str, tgt_path: str) -> Image.Image:
    """Concatenate source and target images side by side (horizontally)."""
    src = Image.open(src_path).convert("RGB")
    tgt = Image.open(tgt_path).convert("RGB")

    # Resize to same height if needed
    h = min(src.height, tgt.height)
    src = src.resize((int(src.width * h / src.height), h))
    tgt = tgt.resize((int(tgt.width * h / tgt.height), h))

    new_w = src.width + tgt.width
    new_img = Image.new("RGB", (new_w, h))
    new_img.paste(src, (0, 0))
    new_img.paste(tgt, (src.width, 0))
    return new_img


def sample_and_save(df: pd.DataFrame, outdir: Path, bins: int, samples_per_bin: int):
    outdir.mkdir(parents=True, exist_ok=True)

    # Shared bin edges
    all_lpips = df["lpips"].to_numpy()
    bin_edges = np.linspace(all_lpips.min(), all_lpips.max(), bins + 1)

    # Assign bin index
    df = df.copy()
    df["bin"] = pd.cut(df["lpips"], bins=bin_edges, labels=False, include_lowest=True)

    for b in range(bins):
        bin_df = df[df["bin"] == b]
        if bin_df.empty:
            print(f"[Bin {b:02d}] empty.")
            continue

        bin_dir = outdir / f"bin_{b:02d}"
        bin_dir.mkdir(exist_ok=True)

        chosen = bin_df.sample(n=min(samples_per_bin, len(bin_df)), replace=False, random_state=42)

        count = 0
        for _, row in chosen.iterrows():
            lp_str = f"{row['lpips']:.4f}"
            out_name = f"{row['data_id']}_lpips{lp_str}.png"
            try:
                combined = concat_side_by_side(row["source"], row["target"])
                combined.save(bin_dir / out_name)
                count += 1
            except FileNotFoundError as e:
                print(f"[Warn] Missing file for {row['data_id']}: {e}")

        print(f"[Bin {b:02d}] Saved {count} combined images to {bin_dir}.")

    # ---- Save histogram ----
    plt.figure(figsize=(8, 6))
    for split, g in df.groupby("split"):
        plt.hist(
            g["lpips"],
            bins=bin_edges,
            alpha=0.6,
            label=split,
            histtype="stepfilled",
        )
    plt.xlabel("LPIPS")
    plt.ylabel("Count")
    plt.title("LPIPS Histogram by Split")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    hist_path = outdir / "lpips_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[Info] Histogram saved to {hist_path}")


def main():
    ap = argparse.ArgumentParser(description="Sample images from LPIPS bins, save side-by-side pairs, plus histogram.")
    ap.add_argument("--json", required=True, type=str, help="Path to dataset.json")
    ap.add_argument("--outdir", required=True, type=str, help="Output directory for sampled bins + histogram")
    ap.add_argument("--bins", type=int, default=10, help="Number of LPIPS bins")
    ap.add_argument("--samples", type=int, default=5, help="Number of pairs to sample per bin")
    args = ap.parse_args()

    df = load_manifest(Path(args.json))
    sample_and_save(df, Path(args.outdir), args.bins, args.samples)


if __name__ == "__main__":
    main()

