#!/bin/bash

python inference_i2t.py \
  --model_path "MeissonFlow/Meissonic" \
  --transformer_path "path/to/transformer" \
  --image_path_or_dir "assets/demo.jpg" \
  --resolution 512 \
  --steps 32 \
  --cfg 9.0 \
  --device "cuda"