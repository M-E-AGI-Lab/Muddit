#!/bin/bash

python inference_t2i.py \
  --model_path "MeissonFlow/Meissonic" \
  --transformer_path "path/to/transformer" \
  --resolution 512 \
  --steps 64 \
  --cfg 9.0 \
  --device "cuda"