import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import random
import numpy as np


def pad_and_crop(img, target_size, top=None, left=None):
    """Pad first (if needed), then random crop to target_size (H, W).
    Works with [C,H,W] or [N,C,H,W].
    """
    th, tw = target_size

    if img.dim() == 3:  # [C,H,W]
        img = img.unsqueeze(0)  # -> [1,C,H,W]
        squeeze_back = True
    else:
        squeeze_back = False

    n, c, h, w = img.shape

    # ---- Step 1: pad if smaller ----
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

    # ---- Step 2: random crop ----
    if top is None:
        top = random.randint(0, h - th) if h > th else 0
    if left is None:
        left = random.randint(0, w - tw) if w > tw else 0
    img = img[..., top:top+th, left:left+tw]  # [N,C,th,tw]
    
    if squeeze_back:
        img = img.squeeze(0)  # return [C,H,W]

    return img, top, left


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None):
    #def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]
        
        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(input_img)
        nopad_mask = torch.ones_like(img_t)
        #img_t = F.resize(img_t, self.image_size)
        img_t, top, left = pad_and_crop(img_t, self.image_size)
        nopad_mask, _, _ = pad_and_crop(nopad_mask, self.image_size, top, left)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])



        output_t = F.to_tensor(output_img)
        output_t, _, _ = pad_and_crop(output_t, self.image_size, top, left)
        #output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            ref_t, _, _ = pad_and_crop(ref_t, self.image_size)
            #ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)
        nopad_mask = nopad_mask.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "scene_id": scene_id,
            "nopad_mask": nopad_mask
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out

class PairedDatasetCus(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None, mulref=False, nv=1, useRender=False, stich=False, select=False):
    #def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        if split == "train" and select:
            ks = list(self.data.keys())
            for k in ks:
                if self.data[k]["lpips"] > 0.3:
                    del self.data[k]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.img_ids_sorted = sorted(self.img_ids)
        self.mulref = mulref
        self.nv = nv
        self.useRender = useRender
        self.stich = stich
        self.select = select

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]
        
        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(input_img)
        #img_t = F.resize(img_t, self.image_size)
        nopad_mask = torch.ones_like(img_t)
        img_t, top, left = pad_and_crop(img_t, self.image_size)
        nopad_mask, _, _ = pad_and_crop(nopad_mask, self.image_size, top, left)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t, _, _ = pad_and_crop(output_t, self.image_size, top, left)
        #output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if not self.mulref:
            # get closest id
            # this will cause a bug when there is only one target
            i = self.img_ids_sorted.index(img_id) # O(n) search for exact match
            if i==0:
                neighbor_id = self.img_ids_sorted[i+1]
            elif i==len(self.img_ids_sorted)-1:
                neighbor_id = self.img_ids_sorted[i-1]
            else:
                neighbor_id = self.img_ids_sorted[i-1] if random.random() < 0.5 else self.img_ids_sorted[i+1]

            # get ref image path
            path = self.data[img_id]["target_image"]
            parts = path.split("/")
            ref_img = "/".join(parts[:-1] + [neighbor_id+"_target.png"])

            # check if neighbor is valid (same scene prefix)
            scene_id = img_id.split("_val_")[0]
            use_neighbor = neighbor_id is not None and neighbor_id.split("_val_")[0] == scene_id

            if use_neighbor:
                if self.useRender:
                    ref_name = "_source.png"
                else:
                    ref_name = "_target.png"
                ref_img = "/".join(parts[:-1] + [neighbor_id + ref_name])
                ref_t = Image.open(ref_img)
                ref_t = F.to_tensor(ref_t)
                ref_t, _, _ = pad_and_crop(ref_t, self.image_size)
                ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
            else:
                # make a blank normalized image with same shape as img_t
                ref_t = torch.full_like(img_t, -1.0)
        else:
            # --- step 1: find same-scene IDs, exclude source ---
            scene_id = img_id.split("_val_")[0]
            scene_ids = [iid for iid in self.img_ids_sorted if iid.split("_val_")[0] == scene_id and iid != img_id]

            factor = int(self.nv**0.5)

            # --- step 2: sample up to 16 neighbors ---
            chosen_ids = random.sample(scene_ids, k=min(self.nv, len(scene_ids)))

            # --- step 3: pre-fill blanks (16 slots) ---
            refs = torch.full(
                (self.nv, 3, self.image_size[0]//factor, self.image_size[1]//factor),
                -1.0
            )  # [16,C,H/4,W/4]

            # --- step 4: load chosen refs, convert to tensor, batch process ---
            loaded = []
            for nid in chosen_ids:
                if self.useRender:
                    ref_key = "image"
                    ref_name = "_source.png"
                else:
                    ref_key = "target_image"
                    ref_name = "_target.png"
                path = self.data[img_id][ref_key]
                parts = path.split("/")
                ref_img = "/".join(parts[:-1] + [nid + ref_name])
                loaded.append(Image.open(ref_img))

            if loaded:
                ref_ts = [F.to_tensor(img) for img in loaded]          # list of [C,H,W]
                ref_ts = torch.stack(ref_ts, dim=0)                   # [N,C,H,W]

                _, h, w = ref_ts[0].shape 

                # 1. Resize first (downscale)
                small_size = [h // factor, w // factor]
                ref_ts_small = F.resize(ref_ts, small_size)  # [N,C,h,w]

                # 2. Pad/crop at the smaller resolution
                small_size = [self.image_size[0] // factor, self.image_size[1] // factor]
                ref_ts_small, _, _ = pad_and_crop(ref_ts_small, small_size)  

                ref_ts_small = F.normalize(ref_ts_small, mean=[0.5], std=[0.5])   # [N,C,H,W]

                refs[:len(ref_ts_small)] = ref_ts_small               # overwrite blanks

            # --- step 5: tile 16 small refs into 4x4 grid ---
            rows = []
            for i in range(factor):
                row = torch.cat(list(refs[i*factor:(i+1)*factor]), dim=2)   # concat along width
                rows.append(row)
            ref_t = torch.cat(rows, dim=1)                     # concat rows along height
        
        if self.stich:
            # Side-by-side stitching, keep batch dimension
            img_t    = torch.cat([img_t, ref_t], dim=-1).unsqueeze(0)       # (1, C, H, W*2)
            output_t = torch.cat([output_t, ref_t], dim=-1).unsqueeze(0)    # (1, C, H, W*2)
        else:
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)

        nopad_mask = nopad_mask.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "scene_id": scene_id,
            "nopad_mask": nopad_mask.to(torch.bool) 
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out

class PairedDatasetCur(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        height=512,
        width=512,
        tokenizer=None,
        mulref=False,
        select=False,
        nv=1,
        useRender=False,
        stich=False,
        num_buckets=10,
        unlock_schedule=[1, 3, 5, 7, 9, 11, 13, 15, 17, 20],
        old_ratio=0.5,
        use_prev_union=True,
        seed=42,
    ):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        assert split == "train"

        # filter out too-hard samples
        if select:
            ks = list(self.data.keys())
            for k in ks:
                if self.data[k]["lpips"] > 0.6:
                    del self.data[k]

        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.img_ids_sorted = sorted(self.img_ids)
        self.mulref = mulref
        self.nv = int(nv)
        self.useRender = useRender
        self.stich = stich

        # ---------- Curriculum Setup ----------
        self.num_buckets = int(max(1, num_buckets))
        self._rng = np.random.default_rng(seed)
        self._old_ratio = old_ratio
        self._use_prev_union = use_prev_union

        if unlock_schedule is None:
            self.unlock_schedule = list(range(1, self.num_buckets))
        else:
            assert len(unlock_schedule) == self.num_buckets, \
                "unlock_schedule should have num_buckets-1 entries"
            self.unlock_schedule = unlock_schedule

        has_all_lpips = all("lpips" in self.data[k] for k in self.img_ids)
        if not has_all_lpips or len(self.img_ids) < self.num_buckets:
            raise ValueError(
                "Curriculum learning setup failed: all samples must contain 'lpips' "
                "and the number of samples must be >= the number of curriculum buckets."
            )

        lpips_vals = np.array([self.data[k]["lpips"] for k in self.img_ids], dtype=np.float32)
        order = np.argsort(lpips_vals)
        self._sorted_ids = [self.img_ids[i] for i in order]
        self._buckets = np.array_split(np.array(self._sorted_ids, dtype=object), self.num_buckets)

        # Curriculum state
        self.current_epoch = 0
        self._unlocked_bucket = 0
        self._old_pool = None
        self._new_pool = None
        self._uniform_all = False  # <--- NEW
        self._refresh_pools()

    #def __len__(self):
    #    return len(self.img_ids)

    def __len__(self):
        """Length equals total number of samples in currently unlocked buckets."""
        if self._uniform_all:
            return len(self.img_ids)
        # count samples in unlocked buckets up to current index
        unlocked_ids = np.concatenate(self._buckets[: self._unlocked_bucket + 1])
        return len(unlocked_ids)

    def set_epoch(self, epoch: int):
        """Call this once per epoch (DataLoader has a set_epoch hook for this)."""
        self.current_epoch = epoch
        num_unlocked = sum(epoch >= t for t in self.unlock_schedule) + 1
        self._unlocked_bucket = min(num_unlocked - 1, self.num_buckets - 1)

        # once we've reached the final unlock → uniform over all data
        self._uniform_all = (epoch >= self.unlock_schedule[-1])

        self._refresh_pools()

    def _refresh_pools(self):
        b = self._unlocked_bucket
        new_pool = self._buckets[b]
        if b == 0:
            old_pool = None
        else:
            old_pool = np.concatenate(self._buckets[:b]) if self._use_prev_union else self._buckets[b - 1]
        self._new_pool = new_pool
        self._old_pool = old_pool

    def _pick_img_id(self):
        """Sampling policy:
        - Before final unlock: old_ratio from old pools, (1-old_ratio) from current bucket.
        - After final unlock epoch: uniform over entire dataset.
        """
        if self._uniform_all:
            return str(self._rng.choice(self._sorted_ids))

        if self._old_pool is None or len(self._old_pool) == 0:
            return str(self._rng.choice(self._new_pool))
        if self._rng.random() < self._old_ratio:
            return str(self._rng.choice(self._old_pool))
        else:
            return str(self._rng.choice(self._new_pool))

    def __getitem__(self, idx):
        img_id = self._pick_img_id()
        rec = self.data[img_id]
        input_path = rec["image"]
        output_path = rec["target_image"]
        caption = rec["prompt"]

        try:
            input_img = Image.open(input_path)
            output_img = Image.open(output_path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.img_ids))

        # Source
        img_t = F.to_tensor(input_img)
        nopad_mask = torch.ones_like(img_t)
        img_t, top, left = pad_and_crop(img_t, self.image_size)
        nopad_mask, _, _ = pad_and_crop(nopad_mask, self.image_size, top, left)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        # Target
        output_t = F.to_tensor(output_img)
        output_t, _, _ = pad_and_crop(output_t, self.image_size, top, left)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        # Reference
        if not self.mulref:
            i = self.img_ids_sorted.index(img_id)
            if i == 0:
                neighbor_id = self.img_ids_sorted[i + 1]
            elif i == len(self.img_ids_sorted) - 1:
                neighbor_id = self.img_ids_sorted[i - 1]
            else:
                neighbor_id = self.img_ids_sorted[i - 1] if random.random() < 0.5 else self.img_ids_sorted[i + 1]

            parts = self.data[img_id]["target_image"].split("/")
            scene_id = img_id.split("_val_")[0]
            use_neighbor = neighbor_id is not None and neighbor_id.split("_val_")[0] == scene_id

            if use_neighbor:
                ref_name = "_source.png" if self.useRender else "_target.png"
                ref_img_path = "/".join(parts[:-1] + [neighbor_id + ref_name])
                ref_t = Image.open(ref_img_path)
                ref_t = F.to_tensor(ref_t)
                ref_t, _, _ = pad_and_crop(ref_t, self.image_size)
                ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
            else:
                ref_t = torch.full_like(img_t, -1.0)
        else:
            scene_id = img_id.split("_val_")[0]
            scene_ids = [iid for iid in self.img_ids_sorted if iid.split("_val_")[0] == scene_id and iid != img_id]
            factor = int(self.nv ** 0.5)
            chosen_ids = random.sample(scene_ids, k=min(self.nv, len(scene_ids)))

            refs = torch.full(
                (self.nv, 3, self.image_size[0] // factor, self.image_size[1] // factor),
                -1.0
            )
            loaded = []
            for nid in chosen_ids:
                if self.useRender:
                    ref_key = "image"; ref_name = "_source.png"
                else:
                    ref_key = "target_image"; ref_name = "_target.png"
                path = self.data[img_id][ref_key]
                parts = path.split("/")
                rpath = "/".join(parts[:-1] + [nid + ref_name])
                loaded.append(Image.open(rpath))

            if loaded:
                ref_ts = [F.to_tensor(img) for img in loaded]
                ref_ts = torch.stack(ref_ts, dim=0)
                _, h, w = ref_ts[0].shape
                small_size = [h // factor, w // factor]
                ref_ts_small = F.resize(ref_ts, small_size)
                small_size = [self.image_size[0] // factor, self.image_size[1] // factor]
                ref_ts_small, _, _ = pad_and_crop(ref_ts_small, small_size)
                ref_ts_small = F.normalize(ref_ts_small, mean=[0.5], std=[0.5])
                refs[:len(ref_ts_small)] = ref_ts_small

            rows = []
            for i_row in range(factor):
                row = torch.cat(list(refs[i_row * factor:(i_row + 1) * factor]), dim=2)
                rows.append(row)
            ref_t = torch.cat(rows, dim=1)

        # Stitch or Stack
        if self.stich:
            img_t    = torch.cat([img_t, ref_t], dim=-1).unsqueeze(0)
            output_t = torch.cat([output_t, ref_t], dim=-1).unsqueeze(0)
        else:
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)

        nopad_mask = nopad_mask.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "scene_id": img_id.split("_val_")[0],
            "nopad_mask": nopad_mask.to(torch.bool),
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
