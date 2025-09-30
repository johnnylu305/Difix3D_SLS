export NUM_NODES=1
export NUM_GPUS=2
accelerate launch --mixed_precision=bf16 --main_process_port 29503 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_difix_pretrain_mulref.py \
    --output_dir=./outputs_newjson_bs2_pretrain_mulref_src_stich_no0.6/difix/train \
    --dataset_path="../SLS_Fixer_Data/dataset.json" \
    --max_train_steps 30000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=2 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2500 --eval_freq 1000 --viz_freq 500 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 \
    --nv 16 \
    --useRender \
    --stich \
    --select \
