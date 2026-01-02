#!/bin/bash

python inference_i2t.py \
  --model_path "MeissonFlow/Meissonic" \
  --transformer_path "path/to/transformer" \
  --image_path "assets/COCO_val2014_000000000395.jpg" \
  --question "What is the man doing in the image?" \
  --resolution 1024 \
  --steps 32 \
  --cfg 1.0 \
  --device "cuda"