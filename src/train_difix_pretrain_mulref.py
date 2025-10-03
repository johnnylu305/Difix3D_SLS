import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDatasetCus
from loss import gram_loss
from pipeline_difix import DifixPipeline

from pytorch_msssim import ssim
import math
#import matplotlib.pyplot as plt
import cv2
os.makedirs("./output_att", exist_ok=True)
# will hold the three processors so we can read their .last_attn_probs after a forward
REC_PROCS = {"down": None, "mid": None, "up": None}
ATT_HW = {"down": (64, 128), "mid": (8, 16), "up": (64, 128)}

def charbonnier_loss(pred, target, epsilon=1e-6):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + epsilon**2))


def save_avg_attn_map(
    avg_vec: torch.Tensor,     # [B, K], where K = att_h*att_w or M*att_h*att_w
    img: torch.Tensor,         # [B, 1, C, H, W], values in [-1, 1]
    name: str,
    att_h: int, att_w: int,    # attention native resolution
    outdir: str = "./output_att",
    alpha: float = 0.45,       # overlay opacity
):
    os.makedirs(outdir, exist_ok=True)

    # ---- Sanity ----
    if not (isinstance(avg_vec, torch.Tensor) and isinstance(img, torch.Tensor)):
        raise TypeError("avg_vec and img must be torch.Tensors")
    if avg_vec.ndim != 2:
        raise ValueError(f"avg_vec must be [B,K], got {tuple(avg_vec.shape)}")
    if img.ndim != 5 or img.shape[1] != 1:
        raise ValueError(f"img must be [B,1,C,H,W], got {tuple(img.shape)}")

    vec = avg_vec#.detach().to(torch.float32).cpu()   # [B,K]
    im  = img.detach().to(torch.float32).cpu()       # [B,1,C,H,W]

    B, K = vec.shape
    Bb, V, C, H, W = im.shape
    if B != Bb:
        raise ValueError(f"Batch mismatch: avg_vec B={B}, img B={Bb}")
    if V != 1:
        raise ValueError(f"Expected V=1, got V={V}")

    att_hw = att_h * att_w
    if (K != att_hw) and (K % att_hw != 0):
        raise ValueError(f"K={K} not compatible with att_h*att_w={att_hw}")

    # [-1,1] -> [0,255] uint8 (RGB), then BGR for OpenCV
    def chw_to_bgr_u8(tCHW: torch.Tensor) -> np.ndarray:
        t = tCHW
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] > 3:
            t = t[:3]
        t01 = (t * 0.5 + 0.5).clamp(0, 1)
        rgb = (t01.numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)  # HWC
        return rgb[..., ::-1]  # RGB->BGR

    for b in range(B):
        # Build native attention [1,1,att_h,att_w]
        if K == att_hw:
            att_native = vec[b].view(1, 1, att_h, att_w)
        else:
            M = K // att_hw
            chunks = [vec[b, m*att_hw:(m+1)*att_hw].view(1,1,att_h,att_w) for m in range(M)]
            att_native = torch.mean(torch.stack(chunks, 0), dim=0)

        # Upsample to (H,W)
        att_up = F.interpolate(att_native, size=(H, W), mode="bilinear", align_corners=False)[0,0]  # [H,W]

        # Normalize to [0,1]
        a_min, a_max = float(att_up.min()), float(att_up.max())
        att01 = (att_up - a_min) / (a_max - a_min + 1e-8)

        # Colorize -> heatmap
        att_u8   = (att01.numpy() * 255.0).astype(np.uint8)         # [H,W]
        heat_bgr = cv2.applyColorMap(att_u8, cv2.COLORMAP_JET)      # [H,W,3] uint8

        # Save heatmap
        #heat_path = os.path.join(outdir, f"{name}_b{b}_heat.png")
        #cv2.imwrite(heat_path, heat_bgr)

        # Overlay with image (use view v=0)
        img_bgr = chw_to_bgr_u8(im[b, 0])                           # [H,W,3] uint8 (BGR)
        over    = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)

        #over_path = os.path.join(outdir, f"{name}_b{b}_overlay.png")
        #cv2.imwrite(over_path, over)

        #img_path = os.path.join(outdir, f"{name}_b{b}_img.png")
        #cv2.imwrite(img_path, img_bgr)

        #print(f"[attn] saved {heat_path} and {over_path}")

        # Simple vertical stack: [original; heatmap; overlay]
        combined = np.vstack([heat_bgr, img_bgr, over])        # [3H, W, 3]

        stack_path = os.path.join(outdir, f"{name}_b{b}_stack.png")
        cv2.imwrite(stack_path, combined)
        print(f"[attn] saved {stack_path}")


class RecordingAttnProcessor:
    """
    Callable processor compatible with diffusers' Attention.set_processor().
    Computes attention the standard way and stores probs in .last_attn_probs.
    """
    def __init__(self, tag: str):
        self.tag = tag
        self.last_attn_probs = None  # [B, H, Q, K]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):

        # hidden_states.shape: (B, Seq, D)
        # ex: Seq = H // 8 x W // 8
        # Projections (same as AttnProcessor2_0 pattern)
        q = attn.to_q(hidden_states)
        k_src = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        k = attn.to_k(k_src)
        v = attn.to_v(k_src)
        
        # (B, Seq, D)
        bsz, q_len, dim = q.shape
        # multi-head attention
        # ex: 5
        heads = attn.heads
        # ex: 64 = 320 // 5 
        head_dim = dim // heads
        scale = getattr(attn, "scale", 1.0 / math.sqrt(head_dim))

        def _shape(x):
            # [B, T, D] -> [B, H, T, Dh]
            return x.view(bsz, -1, heads, head_dim).transpose(1, 2)

        q = _shape(q)
        k = _shape(k)
        v = _shape(v)

        # Scores and probs
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,Q,K]
        if attention_mask is not None:
            scores = scores + attention_mask  # mask is bias
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)  # [B,H,Q,K]
        # ex: Q = H//8 x W//8
        # ex: K = H//8 x W//8
        #self.last_attn_probs = probs.detach()

        # === Average attention map ===
        # mean over heads and over queries -> [B, K]
        avg_attn_vec = probs.mean((1, 2))          # [B, K]
        self.avg_attn_vec = avg_attn_vec.detach().to(torch.float32).cpu() #.to(torch.float32)
        #print("h", torch.unique(hidden_states))
        #print("q", torch.unique(q))
        #print("k", torch.unique(k))
        #print("s", torch.unique(scores))
        #print("probs", torch.unique(probs))
        #print("avg", torch.unique(avg_attn_vec))
        #print("avg2", torch.unique(self.avg_attn_vec))
        # Weighted sum (standard attention output)
        # return for mv_net training
        out = torch.matmul(probs, v)  # [B,H,Q,Dh]
        out = out.transpose(1, 2).reshape(bsz, q_len, heads * head_dim)  # [B,Q,D]
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor in range [-1,1] to uint8 [0,255].
    Accepts CHW or BCHW tensors.
    """
    img = img.clamp(-1, 1)         # step 1: clip
    img = (img + 1.0) / 2.0        # step 2: rescale to [0,1]
    img = (img * 255).round()      # step 3: scale to [0,255]
    return img.to(torch.uint8)


def load_pipe_weights_into_model(pipe, model, report: bool = True):
    """
    Load weights from a DifixPipeline (pipe) into a Difix model (model).

    Args:
        pipe: DifixPipeline instance (HuggingFace Diffusers style).
        model: Difix model instance (your training version).
        report: If True, print a summary of loaded/missing/unexpected keys.
    """

    results = {}

    def load_component(dst_module, src_module, name):
        dst_sd = dst_module.state_dict()
        src_sd = src_module.state_dict()

        # filter out mismatched shapes
        src_sd = {
            k: v for k, v in src_sd.items()
            if k in dst_sd and v.shape == dst_sd[k].shape
        }

        missing, unexpected = dst_module.load_state_dict(src_sd, strict=False)
        results[name] = {"missing": missing, "unexpected": unexpected}

        if report:
            if not missing:
                status = "✅ OK"
            else:
                status = "❌ NOT OK"
            print(f"{name}: {dst_module.__class__.__name__} "
                  f"(params: {len(dst_sd)}) --> {status}")
            if missing:
                print(f"   Missing ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"   Unexpected ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    # UNet
    load_component(model.unet, pipe.unet, "unet")

    # VAE
    load_component(model.vae, pipe.vae, "vae")

    # Text encoder
    load_component(model.text_encoder, pipe.text_encoder, "text_encoder")

    # Copy over tokenizer / scheduler (not modules)
    if hasattr(model, "tokenizer"):
        model.tokenizer = pipe.tokenizer
    if hasattr(model, "scheduler"):
        model.scheduler = pipe.scheduler

    return results


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae, 
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )

    # get attention
    # --- attach processors to exactly one down/mid/up self-attn (attn1) ---
    unet_ref = net_difix.unet  # keep a non-wrapped reference for name matching (before accelerator.prepare)
    down_set = mid_set = up_set = False
    for name, module in unet_ref.named_modules():
        if hasattr(module, "attn1"): #1"):
            #print(name, module)
            if (not down_set) and ("down_blocks.0.attentions.1.transformer_blocks.0" in name):
                REC_PROCS["down"] = RecordingAttnProcessor("down")
                module.attn1.set_processor(REC_PROCS["down"])
                down_set = True
            elif (not mid_set) and ("mid_block.attentions.0.transformer_blocks.0" in name):
                REC_PROCS["mid"] = RecordingAttnProcessor("mid")
                module.attn1.set_processor(REC_PROCS["mid"])
                mid_set = True
            elif (not up_set) and ("up_blocks.3.attentions.2.transformer_blocks.0" in name):
                REC_PROCS["up"] = RecordingAttnProcessor("up")
                module.attn1.set_processor(REC_PROCS["up"])
                up_set = True
    
    # Original difix
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")
    load_pipe_weights_into_model(pipe, net_difix, report=True)
    del pipe
    net_difix.set_train()


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()

    net_lpips.requires_grad_(False)
    
    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    layers_to_opt += list(net_difix.unet.parameters())
   
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_difix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    dataset_train = PairedDatasetCus(dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer, mulref=True, nv=args.nv, useRender=args.useRender, stich=args.stich, select=args.select)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDatasetCus(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer, mulref=True, nv=args.nv, useRender=args.useRender, stich=args.stich, select=args.select)
    random.Random(42).shuffle(dataset_val.img_ids)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare everything with our `accelerator`.
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    # renorm with image net statistics
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

        progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
            disable=not accelerator.is_local_main_process,)


    # compute validation set L2, LPIPS
    initial_val = True
    if initial_val and accelerator.is_main_process:
        logs = {}
        l_l2, l_lpips, src_l_l2, src_l_lpips = [], [], [], []
        l_psnr, src_l_psnr = [], []
        #log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": [], "sample/results": []}
        log_dict = {"sample/results": []}
        seen = {}
        for step, batch_val in enumerate(dl_val):
            #print(step)
            if step >= args.num_samples_eval:
                break
            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
            nopad_mask = batch_val["nopad_mask"].to(accelerator.device, dtype=weight_dtype)
            B, V, C, H, W = x_src.shape
            assert B == 1, "Use batch size 1 for eval."
            with torch.no_grad():
                # forward pass
                x_tgt_pred = accelerator.unwrap_model(net_difix)(x_src, prompt_tokens=batch_val["input_ids"].cuda())
                
                if batch_val["scene_id"][0] not in seen:
                    seen[batch_val["scene_id"][0]] = 0

                scene_flag = True
                for s in ["crab", "yoda", "android", "patio_first", "statue", "mountain", "corner"]:
                    if s in batch_val["scene_id"][0]:
                        scene_flag = False

                if step>250 and scene_flag and seen[batch_val["scene_id"][0]] < 12:
                    seen[batch_val["scene_id"][0]] += 1
                #if step % 2 == 0: #10 == 0:
                    #log_dict["sample/source"].append(wandb.Image(to_uint8(rearrange(x_src, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                    #log_dict["sample/target"].append(wandb.Image(to_uint8(rearrange(x_tgt, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                    #log_dict["sample/model_output"].append(wandb.Image(to_uint8(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                    if args.stich:
                        # b v c h w
                        # src_tar | src | src_output 
                        #         | ref | ref_output
                        b, v, c, h, w = x_src.shape
                        w2 = w // 2
                        # split halves
                        src, ref = x_src[..., :w2], x_src[..., w2:]
                        src_tar = x_tgt[..., :w2]
                        src_output, ref_output = x_tgt_pred[..., :w2], x_tgt_pred[..., w2:]

                        # row1: src_tar | src | src_output
                        row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w2)
                        # row2: blank | ref | ref_output
                        blank = -torch.ones_like(ref)
                        row2 = torch.cat([blank, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w2)
                        # 2×3 grid
                        grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w2)
                    else:
                        # b v c h w
                        # src_tar | src | src_output 
                        #         | ref | ref_output
                        b, v, c, h, w = x_src.shape
                        # split halves
                        src, ref = x_src[:, 0:1, :, :, :], x_src[:, 1:2, :, :, :]
                        src_tar = x_tgt[:, 0:1, :, :, :]
                        src_output, ref_output = x_tgt_pred[:, 0:1, :, :, :], x_tgt_pred[:, 1:2, :, :, :]
                        # row1: src_tar | src | src_output
                        row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w2)
                        # row2: blank | ref | ref_output
                        blank = -torch.ones_like(ref)
                        row2 = torch.cat([blank, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w2)
                        # 2×3 grid
                        grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w2)

                    log_dict["sample/results"].append(wandb.Image(to_uint8(rearrange(grid, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/results'])}"))
                # b v c h w -> b c h w
                x_tgt = x_tgt[:, 0] # take the input view
                x_tgt_pred = x_tgt_pred[:, 0] # take the input view
                x_src = x_src[:, 0]
                # compute the reconstruction losses
                ys, xs = torch.where(nopad_mask[0, 0, 0, :, :])
                y0, y1 = ys.min().item(), ys.max().item() + 1
                x0, x1 = xs.min().item(), xs.max().item() + 1
                #plt.imshow(np.transpose(to_uint8(x_tgt_pred[...,:w//2][...,y0:y1,x0:x1])[0].detach().cpu().numpy(), (1, 2, 0)))
                #plt.show()
                #plt.imshow(np.transpose(to_uint8(x_tgt[...,:w//2][...,y0:y1,x0:x1])[0].detach().cpu().numpy(), (1, 2, 0)))
                #plt.show()
                if args.stich:
                    b, c, h, w = x_tgt_pred.shape
                    loss_l2 = F.mse_loss((x_tgt_pred[...,:w//2][...,y0:y1,x0:x1]).float(), (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float(), reduction="mean")
                    loss_lpips = net_lpips((x_tgt_pred[...,:w//2][...,y0:y1,x0:x1]).float(), (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float()).mean()
                    src_loss_l2 = F.mse_loss((x_src[...,:w//2][...,y0:y1,x0:x1]).float(), (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float(), reduction="mean")
                    src_loss_lpips = net_lpips((x_src[...,:w//2][...,y0:y1,x0:x1]).float(), (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float()).mean()

                    # ---- PSNR computation (batch-aware) ----
                    # -1~1 to 0~1
                    x_tgt = x_tgt*0.5+0.5
                    x_tgt_pred = x_tgt_pred*0.5+0.5
                    x_src = x_src*0.5+0.5
                    mse_per_batch = F.mse_loss(
                        (x_tgt_pred[...,:w//2][...,y0:y1,x0:x1]).float(),
                        (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float(),
                        reduction="none"
                    ).view(b, -1).mean(dim=1)  # shape: (B,)
                    psnr_per_batch = 10 * torch.log10(1.0 / mse_per_batch)
                    psnr = psnr_per_batch.mean().item()

                    src_mse_per_batch = F.mse_loss(
                        (x_src[...,:w//2][...,y0:y1,x0:x1]).float(),
                        (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float(),
                        reduction="none"
                    ).view(b, -1).mean(dim=1)
                    src_psnr_per_batch = 10 * torch.log10(1.0 / src_mse_per_batch)
                    src_psnr = src_psnr_per_batch.mean().item()
                else:
                    loss_l2 = F.mse_loss((x_tgt_pred[...,y0:y1,x0:x1]).float(), (x_tgt[...,y0:y1,x0:x1]).float(), reduction="mean")
                    loss_lpips = net_lpips((x_tgt_pred[...,y0:y1,x0:x1]).float(), (x_tgt[...,y0:y1,x0:x1]).float()).mean()
                    src_loss_l2 = F.mse_loss((x_src[...,y0:y1,x0:x1]).float(), (x_tgt[...,y0:y1,x0:x1]).float(), reduction="mean")
                    src_loss_lpips = net_lpips((x_src[...,y0:y1,x0:x1]).float(), (x_tgt[...,y0:y1,x0:x1]).float()).mean()

                    # ---- PSNR computation (batch-aware) ----
                    # -1~1 to 0~1
                    x_tgt = x_tgt*0.5+0.5
                    x_tgt_pred = x_tgt_pred*0.5+0.5
                    x_src = x_src*0.5+0.5
                    mse_per_batch = F.mse_loss(
                        (x_tgt_pred[...,y0:y1,x0:x1]).float(),
                        (x_tgt[...,y0:y1,x0:x1]).float(),
                        reduction="none"
                    ).view(b, -1).mean(dim=1)  # shape: (B,)
                    psnr_per_batch = 10 * torch.log10(1.0 / mse_per_batch)
                    psnr = psnr_per_batch.mean().item()

                    src_mse_per_batch = F.mse_loss(
                        (x_src[...,y0:y1,x0:x1]).float(),
                        (x_tgt[...,y0:y1,x0:x1]).float(),
                        reduction="none"
                    ).view(b, -1).mean(dim=1)
                    src_psnr_per_batch = 10 * torch.log10(1.0 / src_mse_per_batch)
                    src_psnr = src_psnr_per_batch.mean().item()

                l_l2.append(loss_l2.item())
                l_lpips.append(loss_lpips.item())
                src_l_l2.append(src_loss_l2.item())
                src_l_lpips.append(src_loss_lpips.item())
                l_psnr.append(psnr)
                src_l_psnr.append(src_psnr)

        logs["val/l2"] = np.mean(l_l2)
        logs["val/lpips"] = np.mean(l_lpips)
        logs["val/psnr"] = np.mean(l_psnr)
        logs["val/src_l2"] = np.mean(src_l_l2)
        logs["val/src_lpips"] = np.mean(src_l_lpips)
        logs["val/src_psnr"] = np.mean(src_l_psnr)
        for k in log_dict:
            logs[k] = log_dict[k]
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.log(logs, step=0)

    # start the training loop
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                nopad_mask = batch["nopad_mask"]
            
                B, V, C, H, W = x_src.shape

                # forward pass
                x_tgt_pred = net_difix(x_src, prompt_tokens=batch["input_ids"])       
               
                # get attention
                if accelerator.is_main_process and (global_step % 100 == 0):
                    _, V, _, _, _ = x_src.shape  # number of views in this batch
                    for tag in ("down", "mid", "up"):
                        proc = REC_PROCS[tag]
                        h, w = ATT_HW[tag]
                        #if proc is not None and proc.last_attn_probs is not None:
                        save_avg_attn_map(proc.avg_attn_vec, x_src, tag+"_"+str(global_step), h, w)

                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                         
                # Reconstruction loss
                # x_tgt: (bv) c h w 
                #x_tgt_nonref = x_tgt[::2]
                #plt.imshow(np.transpose(np.array(x_tgt_nonref[0].detach().cpu()), (1, 2, 0)))
                #plt.imshow(np.transpose(np.array(x_tgt_nonref[1].detach().cpu()), (1, 2, 0)))
                #plt.show()
                if args.useRender:
                    if args.stich:
                        b, c, h, w = x_tgt_pred.shape
                        pred = x_tgt_pred[..., :w//2]
                        tgt  = x_tgt[..., :w//2]
                    else:
                        pred = x_tgt_pred[::2]
                        tgt  = x_tgt[::2]
                else:
                    pred = x_tgt_pred
                    tgt  = x_tgt

                # ---- apply nopad_mask ----
                nopad_mask = rearrange(nopad_mask, 'b v c h w -> (b v) c h w')
                # TODO: reference does not support mask...
                pred = pred * nopad_mask
                tgt  = tgt * nopad_mask

                #plt.imshow(np.transpose(to_uint8(pred)[0].detach().cpu().numpy(), (1, 2, 0)))
                #plt.show()
                #plt.imshow(np.transpose(to_uint8(tgt)[0].detach().cpu().numpy(), (1, 2, 0)))
                #plt.show()

                # ---- compute losses ----
                loss_l2 = F.mse_loss(pred.float(), tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(pred.float(), tgt.float()).mean() * args.lambda_lpips
                #loss_ssim = 1 - ssim(pred.float(), tgt.float(), data_range=1.0, size_average=True)
                #loss_charb = charbonnier_loss(pred.float(), tgt.float())
                #loss = loss_charb + loss_lpips 
                loss = loss_l2 + loss_lpips
                
                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        # ---- renormalize ----
                        x_tgt_pred_renorm = t_vgg_renorm(pred * 0.5 + 0.5)
                        x_tgt_renorm = t_vgg_renorm(tgt * 0.5 + 0.5)

                        # ---- random crop ----
                        crop_h, crop_w = 400, 400
                        top = random.randint(0, H - crop_h)
                        left = random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        x_tgt_renorm      = crop(x_tgt_renorm, top, left, crop_h, crop_w)

                        # ---- gram loss ----
                        loss_gram = gram_loss(
                            x_tgt_pred_renorm.to(weight_dtype),
                            x_tgt_renorm.to(weight_dtype),
                            net_vgg
                        ) * args.lambda_gram
                        # loss = loss + loss_gram
                    else:
                        loss_gram = torch.tensor(0.0, dtype=weight_dtype, device=x_tgt_pred.device)                    

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        #log_dict = {
                        #    "train/source": [wandb.Image(to_uint8(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
                        #    "train/target": [wandb.Image(to_uint8(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
                        #    "train/model_output": [wandb.Image(to_uint8(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
                        #}
                        if args.stich:
                            # b v c h w
                            # src_tar | src | src_output 
                            #         | ref | ref_output
                            b, v, c, h, w = x_src.shape
                            w2 = w // 2
                            # split halves
                            src, ref = x_src[..., :w2], x_src[..., w2:]
                            src_tar = x_tgt[..., :w2]
                            src_output, ref_output = x_tgt_pred[..., :w2], x_tgt_pred[..., w2:]

                            # row1: src_tar | src | src_output
                            row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w2)
                            # row2: blank | ref | ref_output
                            blank = -torch.ones_like(ref)
                            row2 = torch.cat([blank, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w2)
                            # 2×3 grid
                            grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w2)
                        else:
                            # b v c h w
                            # src_tar | src | src_output 
                            #         | ref | ref_output
                            b, v, c, h, w = x_src.shape
                            # split halves
                            src, ref = x_src[:, 0:1, :, :, :], x_src[:, 1:2, :, :, :]
                            src_tar = x_tgt[:, 0:1, :, :, :]
                            src_output, ref_output = x_tgt_pred[:, 0:1, :, :, :], x_tgt_pred[:, 1:2, :, :, :]
                            # row1: src_tar | src | src_output
                            row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w2)
                            # row2: blank | ref | ref_output
                            blank = -torch.ones_like(ref)
                            row2 = torch.cat([blank, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w2)
                            # 2×3 grid
                            grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w2)
                        log_dict = {
                            "train/results": [wandb.Image(to_uint8(rearrange(grid, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_difix).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)
                    
                    # compute validation set L2, LPIPS
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        l_psnr = []
                        #log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": [], "sample/results": []}
                        log_dict = {"sample/results": []}
                        seen = {}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            nopad_mask = batch_val["nopad_mask"].to(accelerator.device, dtype=weight_dtype)
                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_difix)(x_src, prompt_tokens=batch_val["input_ids"].cuda())                                

                                if batch_val["scene_id"][0] not in seen:
                                    seen[batch_val["scene_id"][0]] = 0
                                
                                scene_flag = True
                                for s in ["crab", "yoda", "android", "patio_first", "statue", "mountain", "corner"]:
                                    if s in batch_val["scene_id"][0]:
                                        scene_flag = False

                                if step>250 and scene_flag and seen[batch_val["scene_id"][0]] < 12:
                                    #print(batch_val["scene_id"][0])
                                    seen[batch_val["scene_id"][0]] += 1
                                #if step % 2 == 0: #10 == 0:
                                    #log_dict["sample/source"].append(wandb.Image(to_uint8(rearrange(x_src, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                                    #log_dict["sample/target"].append(wandb.Image(to_uint8(rearrange(x_tgt, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                                    #log_dict["sample/model_output"].append(wandb.Image(to_uint8(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                                    if args.stich:
                                        # b v c h w
                                        # src_tar | src | src_output 
                                        #         | ref | ref_output
                                        b, v, c, h, w = x_src.shape
                                        w2 = w // 2
                                        # split halves
                                        src, ref = x_src[..., :w2], x_src[..., w2:]
                                        src_tar = x_tgt[..., :w2]
                                        src_output, ref_output = x_tgt_pred[..., :w2], x_tgt_pred[..., w2:]

                                        # row1: src_tar | src | src_output
                                        row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w2)
                                        # row2: blank | ref | ref_output
                                        blank = -torch.ones_like(ref)
                                        row2 = torch.cat([blank, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w2)
                                        # 2×3 grid
                                        grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w2)
                                    else:
                                        # b v c h w
                                        # src_tar | src | src_output 
                                        #         | ref | ref_output
                                        b, v, c, h, w = x_src.shape
                                        # split halves
                                        src, ref = x_src[:, 0:1, :, :, :], x_src[:, 1:2, :, :, :]
                                        src_tar = x_tgt[:, 0:1, :, :, :]
                                        src_output, ref_output = x_tgt_pred[:, 0:1, :, :, :], x_tgt_pred[:, 1:2, :, :, :]
                                        # row1: src_tar | src | src_output
                                        row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w2)
                                        # row2: blank | ref | ref_output
                                        blank = -torch.ones_like(ref)
                                        row2 = torch.cat([blank, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w2)
                                        # 2×3 grid
                                        grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w2)
 
                                    log_dict["sample/results"].append(wandb.Image(to_uint8(rearrange(grid, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/results'])}"))
                                # b v c h w -> b c h w
                                x_tgt = x_tgt[:, 0] # take the input view
                                x_tgt_pred = x_tgt_pred[:, 0] # take the input view
                                # compute the reconstruction losses
                                ys, xs = torch.where(nopad_mask[0, 0, 0, :, :])
                                y0, y1 = ys.min().item(), ys.max().item() + 1
                                x0, x1 = xs.min().item(), xs.max().item() + 1
                                #plt.imshow(np.transpose(to_uint8(x_tgt_pred[...,:w//2][...,y0:y1,x0:x1])[0].detach().cpu().numpy(), (1, 2, 0)))
                                #plt.show()
                                #plt.imshow(np.transpose(to_uint8(x_tgt[...,:w//2][...,y0:y1,x0:x1])[0].detach().cpu().numpy(), (1, 2, 0)))
                                #plt.show()
                                if args.stich:
                                    b, c, h, w = x_tgt_pred.shape
                                    loss_l2 = F.mse_loss((x_tgt_pred[...,:w//2][...,y0:y1,x0:x1]).float(), (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float(), reduction="mean")
                                    loss_lpips = net_lpips((x_tgt_pred[...,:w//2][...,y0:y1,x0:x1]).float(), (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float()).mean()
                                    # ---- PSNR computation (batch-aware) ----
                                    x_tgt_pred = x_tgt_pred*0.5+0.5
                                    x_tgt = x_tgt*0.5+0.5
                                    mse_per_batch = F.mse_loss(
                                        (x_tgt_pred[...,:w//2][...,y0:y1,x0:x1]).float(),
                                        (x_tgt[...,:w//2][...,y0:y1,x0:x1]).float(),
                                        reduction="none"
                                    ).view(b, -1).mean(dim=1)  # shape: (B,)
                                    psnr_per_batch = 10 * torch.log10(1.0 / mse_per_batch)
                                    psnr = psnr_per_batch.mean().item()
                                else:
                                    loss_l2 = F.mse_loss((x_tgt_pred[...,y0:y1,x0:x1]).float(), (x_tgt[...,y0:y1,x0:x1]).float(), reduction="mean")
                                    loss_lpips = net_lpips((x_tgt_pred[...,y0:y1,x0:x1]).float(), (x_tgt[...,y0:y1,x0:x1]).float()).mean()
                                    # ---- PSNR computation (batch-aware) ----
                                    x_tgt_pred = x_tgt_pred*0.5+0.5
                                    x_tgt = x_tgt*0.5+0.5
                                    mse_per_batch = F.mse_loss(
                                        (x_tgt_pred[...,y0:y1,x0:x1]).float(),
                                        (x_tgt[...,y0:y1,x0:x1]).float(),
                                        reduction="none"
                                    ).view(b, -1).mean(dim=1)  # shape: (B,)
                                    psnr_per_batch = 10 * torch.log10(1.0 / mse_per_batch)
                                    psnr_per_batch = 10 * torch.log10(1.0 / mse_per_batch)
                                    psnr = psnr_per_batch.mean().item()
                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_psnr.append(psnr)

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/psnr"] = np.mean(l_psnr)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=600, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="johnnylu/DifixSLS", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)#10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
 
    parser.add_argument("--nv", type=int, default=1,
        help="Number of reference view",
    )

    parser.add_argument("--useRender", action="store_true")

    parser.add_argument("--stich", action="store_true")
    parser.add_argument("--select", action="store_true")

    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)
