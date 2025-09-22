import argparse
import json
import random
from pathlib import Path
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="Root directory")
    parser.add_argument("--save", required=True, type=str, help="Save directory")
    args = parser.parse_args()

    root = Path(args.root)
    save_root = Path(args.save)
    save_root.mkdir(parents=True, exist_ok=True)

    entries = {}  # will hold all items before splitting

    data = ["000", "001"]  # adjust if needed
    for d in data:
        data_path = root / d
        if not data_path.is_dir():
            continue

        data_folders = [
            p / (p.name + "-All") / "SLS" / "renders"
            for p in data_path.iterdir()
            if p.is_dir()
        ]

        for s in data_folders:
            for img in s.glob("val*.png"):
                # Load image
                im = Image.open(img)
                w, h = im.size

                # Split into left (target) and right (source)
                left = im.crop((0, 0, w // 2, h))      # (x0, y0, x1, y1)
                right = im.crop((w // 2, 0, w, h))

                # Build save paths: path_to_data/scene_name+img_name+target.png
                scene_name = img.parents[2].name   # the <scene>-All folder
                base_name = img.stem               # image name without extension
                data_id = f"{scene_name}_{base_name}"

                save_left = save_root / f"{data_id}_target.png"
                save_right = save_root / f"{data_id}_source.png"

                # Save (overwrite=True)
                left.save(save_left)
                right.save(save_right)

                # Store entry (source = "image", target = "target_image")
                entries[data_id] = {
                    "image": str(save_right.resolve()),
                    "target_image": str(save_left.resolve()),
                    "ref_image": None,
                    "prompt": "remove degradation"
                }

    # ---- Split into train/test ----
    all_ids = list(entries.keys())
    random.shuffle(all_ids)

    split_idx = int(0.8 * len(all_ids))
    train_ids = all_ids[:split_idx]
    test_ids = all_ids[split_idx:]

    dataset = {
        "train": {k: entries[k] for k in train_ids},
        "test": {k: entries[k] for k in test_ids},
    }

    # ---- Save JSON ----
    json_path = save_root / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"JSON saved to {json_path}")

