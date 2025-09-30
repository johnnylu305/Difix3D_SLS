import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import random


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
        #img_t = F.resize(img_t, self.image_size)
        img_t, top, left = pad_and_crop(img_t, self.image_size)
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

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
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
                if self.data[k]["lpips"] > 0.6:
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
        img_t, top, left = pad_and_crop(img_t, self.image_size)
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

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "scene_id": scene_id
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
