import json
import random
from pathlib import Path

# === edit these ===
IN_JSON  = Path("../dataset.json")
OUT_JSON = Path("../dataset_with_clean.json")
N_DUP    = 5000
SEED     = 42
# ===================

random.seed(SEED)

# load
with IN_JSON.open("r") as f:
    data = json.load(f)

train_items = list(data["train"].items())
num_train = len(train_items)

if num_train == 0:
    raise ValueError("No items found in 'train'.")

# sample entries from train
sampled = random.sample(train_items, min(N_DUP, num_train))

# build new entries
new_entries = {}
for key, entry in sampled:
    e = dict(entry)  # shallow copy
    tgt = e["target_image"]

    e["image"] = tgt
    e["target_image"] = tgt
    # keep ref_image and prompt as is

    new_key = key + "_clean"
    # ensure no collision
    while new_key in data["train"] or new_key in new_entries:
        new_key = new_key + "_dup"
    new_entries[new_key] = e

# add to train
data["train"].update(new_entries)

# save
with OUT_JSON.open("w") as f:
    json.dump(data, f, indent=2)

print(f"Original train size: {num_train}")
print(f"Added {len(new_entries)} '_clean' entries.")
print(f"New train size: {len(data['train'])}")
print(f"Test size unchanged: {len(data.get('test', {}))}")
print(f"Saved to {OUT_JSON}")

