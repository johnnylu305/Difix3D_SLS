# 2e-5
accelerate launch --mixed_precision=bf16 src/train_difix_pretrain_mulref.py \
    --output_dir=./outputs_overfit/difix/train \
    --dataset_path="../SLS_Fixer_Data/small_dataset3.json" \
    --max_train_steps 30000 \
    --resolution=512 --learning_rate 5e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --checkpointing_steps=2500 --eval_freq 1000 --viz_freq 10 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 \
    --nv 4 \
    --useRender \
    --stich \
#    --select \
