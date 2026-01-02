#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

export NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "[INFO] nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc_per_node=${NPROC_PER_NODE}, world_size=${WORLD_SIZE}, master=${MASTER_ADDR}:${MASTER_PORT}"


PYTHONPATH="${PYTHONPATH:-./}" accelerate launch \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --num_processes "${WORLD_SIZE}" \
  train/instruction_tuning.py \
  --output_dir "path/to/output" \
  --train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --text_loss_weight 0.3 \
  --max_grad_norm 10 \
  --pretrained_model_name_or_path "MeissonFlow/Meissonic" \
  --pretrained_transformer_path "path/to/transformer" \
  --text_encoder_architecture 'open_clip' \
  --dataset_type 'MultiSourceVLDataset' \
  --instance_data_dir 'path/to/data' \
  --resolution 1024 \
  --mixed_precision bf16 \
  --use_8bit_adam \
  --lr_scheduler constant \
  --dataloader_num_workers 8 \
  --validation_prompts \
      'a boy' \
      'A serene mountain landscape with towering snow-capped peaks, a crystal-clear blue lake reflecting the mountains, dense pine forests, and a vibrant orange sunrise illuminating the sky.' \
      'A playful golden retriever puppy with a shiny coat, bounding through a meadow filled with colorful wildflowers, under a bright, clear blue sky.' \
      'A bustling city street at night, illuminated by vibrant neon signs in various colors, with busy pedestrians, street vendors, and a light rain creating reflective puddles on the pavement.' \
      'A majestic, medieval castle perched on a rugged cliffside, overlooking a vast, calm ocean at sunset, with the sky painted in hues of pink, orange, and purple.' \
      'An elegant ballerina in a white tutu, dancing gracefully on a grand stage with ornate, gold-trimmed curtains, under a spotlight that casts a soft glow.' \
      'A cozy, rustic log cabin nestled in a snow-covered forest, with smoke rising from the stone chimney, warm lights glowing from the windows, and a path of footprints leading to the front door.' \
      'A Cute Cat' \
      'A Snow Mountain' \
  --validation_images 'path/to/validation_images' \
  --validation_vqa_prompts 'Please describe this image in detail.' \
  --max_train_steps 100000 \
  --checkpointing_steps 500 \
  --validation_steps 50 \
  --logging_steps 10 \
  --report_to 'tensorboard'
