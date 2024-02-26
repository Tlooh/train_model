#!/bin/bash


# 示例数组
gpu_ids=${1:-5,7}

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 --main_process_port 29509 ReFL_rank.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/home/linhaojia/liutao/train_model/prepare_rank_data/ranked_data_10s.json" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=100 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/data/liutao/checkpoints/SD_1-4/rank_offline_10" \
  --grad_scale 0.001 \
  --checkpointing_steps 100 \
  --rank_ids="1,2,3,4,5,6,7,8,9,10"
