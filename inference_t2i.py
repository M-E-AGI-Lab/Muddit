import sys
import argparse
sys.path.append("./")

from src.transformer import SymmetricTransformer2DModel
from src.pipeline import UnifiedPipeline
from src.scheduler import Scheduler

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run Meissonic inference.")
    parser.add_argument("--model_path", type=str, default="MeissonFlow/Meissonic")
    parser.add_argument("--transformer_path", type=str, default="MeissonFlow/Meissonic")
    parser.add_argument("--prompt", type=str, default="A dog on the left and a cat on the right.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--cfg", type=float, default=9.0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    model = SymmetricTransformer2DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
    )
    vq_model = VQModel.from_pretrained(args.model_path, subfolder="vqvae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(args.model_path, subfolder="scheduler")

    pipe = UnifiedPipeline(
        vqvae=vq_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=model,
        scheduler=scheduler,
    )
    pipe.to(args.device)

    negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompts = args.prompt

    with torch.no_grad():
        output = pipe(
            prompt=prompts,
            negative_prompt=negative_prompt,
            height=args.resolution,
            width=args.resolution,
            guidance_scale=args.cfg,
            num_inference_steps=args.steps,
        )

    for i, image in enumerate(output.images):
        image.save(f"{i}.png")


if __name__ == "__main__":
    main()
