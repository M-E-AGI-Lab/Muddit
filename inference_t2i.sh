#!/bin/bash

python inference_t2i.py \
  --model_path "MeissonFlow/Meissonic" \
  --transformer_path "path/to/transformer" \
  --prompt "A girl standing in the middle of the street." \
  --resolution 1024 \
  --steps 64 \
  --cfg 9.0 \
  --device "cuda"