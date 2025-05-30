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
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--cfg", type=float, default=9.0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    model = SymmetricTransformer2DModel.from_pretrained(
        args.transformer_path or args.model_path,
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

    prompts = [
        "A line art portrait showcasing a human figure with flowing, textured strokes",
        "A hyper realistic image of a chimpanzee with a glass-enclosed brain on his head, standing amidst lush, bioluminescent foliage in a vibrant futuristic forest",
        "A samurai in a stylized cyberpunk outfit adorned with intricate steampunk gear and floral accents, his Mandalorian armor gleaming under the backlighting",
        "A translucent, minimalist Porsche 911 GT3RS built from sleek carbon fiber, its aerodynamic body designed in the spirit of '60s Braun and modern Apple minimalism",
        "A realistic photograph of a ramadan tent shaped like a crescent moon under a velvety back sky studded with the milky way",
        "A majestic night sky awash with billowing clouds, sparkling with a million twinkling stars",
        "A portrait of John Lennon, captured in the gritty detail of line art",
        "In a world plunged into an unending darkness, remnants of fading starlight pierce through a heavy, smog-filled sky"
    ]

    output = pipe(
        prompt=prompts,
        negative_prompt=negative_prompt,
        height=args.resolution,
        width=args.resolution,
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        mask_token_embedding=args.transformer_path,
        generator=torch.manual_seed(42)
    )

    for i, image in enumerate(output.images):
        image.save(f"{i}.png")


if __name__ == "__main__":
    main()
