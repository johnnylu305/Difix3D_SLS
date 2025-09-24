import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import random


def pad_and_crop(img, target_size):
    """Pad first (if needed), then random crop to target_size (H, W)."""
    h, w = img.shape[-2:]   # CHW format
    th, tw = target_size

    # ---- Step 1: pad if smaller ----
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        h, w = img.shape[-2:]
    
    # ---- Step 2: random crop (now guaranteed h>=th, w>=tw) ----
    top = random.randint(0, h - th) if h > th else 0
    left = random.randint(0, w - tw) if w > tw else 0
    img = img[..., top:top+th, left:left+tw]

    return img


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
        img_t = pad_and_crop(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = pad_and_crop(output_t, self.image_size)
        #output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            ref_t = pad_and_crop(ref_t, self.image_size)
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
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None):
    #def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.img_ids_sorted = sorted(self.img_ids)

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
        img_t = pad_and_crop(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = pad_and_crop(output_t, self.image_size)
        #output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])
       
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

        if ref_img is not None:
            ref_t = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            ref_t = pad_and_crop(ref_t, self.image_size)
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
