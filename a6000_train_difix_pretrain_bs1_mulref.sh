# lr 2e-5 => 1e-5
#export NUM_NODES=1
#export NUM_GPUS=2
accelerate launch --mixed_precision=bf16 src/train_difix_pretrain_mulref.py \
    --output_dir=./outputs_bs1_pretrain_mulref4/difix/train \
    --dataset_path="../dataset.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2500 --eval_freq 1000 --viz_freq 500 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 \
    --mv_unet \
    --nv 4 \
