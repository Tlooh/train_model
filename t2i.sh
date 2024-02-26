#!/bin/bash

  # --train_data_dir="/media/sdb/liutao/datasets/s2c/prompts/large_data/unique_prompts/simple_prompts_10/s2c_vote_9575.json" \
    # --train_data_dir="/media/sdb/liutao/refl_base/ImageReward/data/refl_data.json" \
      # --train_data_dir="/media/sdb/liutao/datasets/s2c/prompts/data/prompts_120000_filter/processed_all_114959.json" \

# 示例数组
gpu_ids=${1:-4,5}

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 --main_process_port 29501 train_t2i.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/data/liutao/mac8/json/s2c_vote_9575_w_img.json" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/data/liutao/checkpoints/SD_1-4/T2I" \
  --checkpointing_steps 100 \
  --image_column="complex_img" \
  --caption_column="simple"
