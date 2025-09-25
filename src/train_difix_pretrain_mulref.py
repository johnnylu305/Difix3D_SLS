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

import math
import matplotlib.pyplot as plt
os.makedirs("./output_att", exist_ok=True)
# will hold the three processors so we can read their .last_attn_probs after a forward
REC_PROCS = {"down": None} #, "mid": None, "up": None}
ATT_HW = {"down": (128, 64)} #, "mid": , "up": None}

def save_avg_attn_map(avg_vec: torch.Tensor, name: str, h: int, w: int, outdir="./output_att"):
    os.makedirs(outdir, exist_ok=True)

    # ---- sanitize tensor ----
    if not isinstance(avg_vec, torch.Tensor):
        raise TypeError(f"avg_vec must be a torch.Tensor, got {type(avg_vec)}")
    vec = avg_vec.detach().to(torch.float32).cpu()
    if vec.ndim == 2:
        vec = vec[0]  # take first in batch
    elif vec.ndim != 1:
        raise ValueError(f"avg_vec must be 1D or 2D, got shape {tuple(avg_vec.shape)}")

    K = vec.numel()
    hw = h * w

    # ---- reshape: single-view or multi-view ----
    if K == hw:
        att_map = vec.view(h, w)
    elif K % hw == 0:
        V = K // hw
        # average across views if multiple are concatenated
        maps = []
        for v in range(V):
            start, end = v * hw, (v + 1) * hw
            maps.append(vec[start:end].view(h, w))
        att_map = torch.stack(maps, dim=0).mean(dim=0)
    else:
        raise ValueError(f"Cannot reshape K={K} into ({h}x{w}) or its multiples.")

    # ---- normalize to [0,1] for visualization ----
    amin = float(att_map.min())
    amax = float(att_map.max())
    att_vis = (att_map - amin) / (amax - amin + 1e-8)

    # ---- save (no title, no axis, no colorbar, no whitespace) ----
    plt.imshow(att_vis.numpy(), cmap="jet", interpolation="nearest")
    plt.axis("off")
    out_path = os.path.join(outdir, f"{name}_avg.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[save_avg_attn_map] saved: {out_path}")

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
        self.last_attn_probs = probs.detach()

        # === Average attention map ===
        # mean over heads and over queries -> [B, K]
        avg_attn_vec = probs.mean((1, 2))          # [B, K]
        self.avg_attn_vec = avg_attn_vec.detach().to(torch.float32)
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
        if hasattr(module, "attn1"):
            if (not down_set) and ("down_blocks.0" in name):
                REC_PROCS["down"] = RecordingAttnProcessor("down")
                module.attn1.set_processor(REC_PROCS["down"])
                down_set = True
            elif (not mid_set) and ("mid_block" in name):
                REC_PROCS["mid"] = RecordingAttnProcessor("mid")
                module.attn1.set_processor(REC_PROCS["mid"])
                mid_set = True
            elif (not up_set) and ("up_blocks.0" in name):
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

    dataset_train = PairedDatasetCus(dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer, mulref=True, nv=args.nv)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDatasetCus(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer, mulref=True, nv=args.nv)
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

    # start the training loop
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, V, C, H, W = x_src.shape

                # forward pass
                x_tgt_pred = net_difix(x_src, prompt_tokens=batch["input_ids"])       
               
                # get attention
                if accelerator.is_main_process and (global_step % args.viz_freq == 1):
                    _, V, _, _, _ = x_src.shape  # number of views in this batch
                    for tag in ("down",):
                        proc = REC_PROCS[tag]
                        h, w = ATT_HW[tag]
                        if proc is not None and proc.last_attn_probs is not None:
                            save_avg_attn_map(proc.avg_attn_vec, tag+"_"+str(global_step), h, w)

                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                         
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                
                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        
                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                    

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
                        log_dict = {
                            "train/source": [wandb.Image(to_uint8(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(to_uint8(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(to_uint8(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), caption=f"idx={idx}") for idx in range(B)],
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
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_difix)(x_src, prompt_tokens=batch_val["input_ids"].cuda())
                                
                                if step % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(to_uint8(rearrange(x_src, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(to_uint8(rearrange(x_tgt, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(to_uint8(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu()), caption=f"idx={len(log_dict['sample/source'])}"))
                                
                                x_tgt = x_tgt[:, 0] # take the input view
                                x_tgt_pred = x_tgt_pred[:, 0] # take the input view
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
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
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="difix", help="The name of the wandb project to log to.")
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
    parser.add_argument("--num_training_epochs", type=int, default=10)
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
   
    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)
