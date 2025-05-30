import sys
import argparse
sys.path.append("./")

from src.transformer import SymmetricTransformer2DModel
from src.pipeline import UnifiedPipeline
from src.scheduler import Scheduler
from train.trainer_utils import load_images_to_tensor

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run Meissonic inference.")
    parser.add_argument("--model_path", type=str, default="MeissonFlow/Meissonic")
    parser.add_argument("--transformer_path", type=str, default="MeissonFlow/Meissonic")
    parser.add_argument("--image_path_or_dir", type=str, default="./output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=32)
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

    try:
        images = load_images_to_tensor(args.image_path_or_dir, target_size=(args.resolution, args.resolution))
    except:
        images = None

    questions = [
        "Please describe this image.",
    ] * len(images)

    output = pipe(
        prompt=questions,
        image=images,
        height=args.resolution,
        width=args.resolution,
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        mask_token_embedding=args.transformer_path,
    )

    for p in output.prompts:
        print(p)


if __name__ == "__main__":
    main()
