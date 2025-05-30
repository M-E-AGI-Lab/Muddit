# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import math
import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())
import json
import gc

import torch
import torch.nn.functional as F
from torch import nn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

import diffusers.optimization
from diffusers import VQModel

from src.scheduler import Scheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import is_wandb_available
from src.pipeline import UnifiedPipeline
from torchvision.utils import save_image, make_grid
from train.trainer_utils import save_checkpoint
from train.dataset_utils import ImageCaptionLargeDataset, MultiSourceVLDataset
from train.dataset_utils import tokenize_prompt, encode_prompt
from src.transformer import SymmetricTransformer2DModel
from train.trainer_utils import load_images_to_tensor

if is_wandb_available():
    import wandb
    # wandb.login(key="")

logger = get_logger(__name__, log_level="INFO")

import torch._dynamo
torch._dynamo.config.verbose = True

# Optionally suppress errors to fall back to eager execution
torch._dynamo.config.suppress_errors = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_transformer_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained transformer.",
    )
    parser.add_argument(
        "--text_encoder_architecture",
        type=str,
        default="open_clip",
        required=False,
        help="The architecture of the text encoder. One of ['CLIP', 'open_clip', 'flan-t5-base','Qwen2-0.5B','gemini-2b',long_t5_clip','t5_clip']",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        required=False,
        help="The type of the dataset.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--llava_json_path",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--llava_image_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--mmmu_json_path",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--mmmu_image_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--vqa_ann_json_path",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--vqa_image_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--coco_qa_json",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--coco_img_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--gqa_json_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--gqa_image_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--mg_llava_json",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--mg_llava_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--training_from_scratch",
        type=bool,
        default=False,
        required=False
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="muse_training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--text_loss_weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--validation_prompts", type=str, nargs="*")
    parser.add_argument("--validation_vqa_prompts", type=str, default=None)
    parser.add_argument("--validation_images", type=str, default=None)
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--split_vae_encode", type=int, required=False, default=None)
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.", required=False)
    parser.add_argument("--use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--text_encoder_use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument("--text_encoder_lora_r", default=16, type=int)
    parser.add_argument("--text_encoder_lora_alpha", default=32, type=int)
    parser.add_argument("--text_encoder_lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--image_to_text_only", action="store_true")
    parser.add_argument("--image_key", type=str, required=False)
    parser.add_argument("--prompt_key", type=str, required=False)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--prompt_prefix", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    if args.instance_data_dir is not None:
        if not os.path.exists(args.instance_data_dir):
            raise ValueError(f"Does not exist: `--args.instance_data_dir` {args.instance_data_dir}")

    return args

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        accelerator.init_trackers("meissonic", config=vars(copy.deepcopy(args)))

    if args.seed is not None:
        set_seed(args.seed)

    if args.text_encoder_architecture == "open_clip":
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", variant=args.variant
        )
        tokenizer_2 = None
        text_encoder_2 = None
        
        text_encoder.requires_grad_(False)
    elif args.text_encoder_architecture == "t5_clip":
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", variant=args.variant
        )
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
        )   
        
        tokenizer_2 = T5Tokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", variant=args.variant,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant=args.variant,
        )
        
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    else:
        raise ValueError(f"Unknown text encoder architecture: {args.text_encoder_architecture}")
    
    vq_model = VQModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vqvae", revision=args.revision, variant=args.variant
    )
    vq_model.requires_grad_(False)

    model = SymmetricTransformer2DModel.from_pretrained(
        args.pretrained_transformer_path, 
        subfolder="transformer", 
        low_cpu_mem_usage=False, 
        device_map=None
    )
    
    if model.config.tokenizer_vocab_size is None:
        if args.text_encoder_architecture == "open_clip":
            model.register_to_config(tokenizer_vocab_size=len(tokenizer))
            # model.config.tokenizer_vocab_size = len(tokenizer)    # We exclude the mask token in the predicted logits
        elif args.text_encoder_architecture == "t5_clip":
            model.register_to_config(tokenizer_vocab_size=len(tokenizer_2))
            # model.config.tokenizer_vocab_size = len(tokenizer_2)      # We don't need to add new token
            if model.adapter is None:
                raise ValueError(f"The MMDiT must has adapter if you want to use t5_clip mode!!!")
        else:
            raise ValueError(f"Unknown text encoder architecture!")
    
        print(f"model's tokenizer vocab size is {model.config.tokenizer_vocab_size}")
        model.text_decoder = nn.Sequential(
            nn.LayerNorm(model.inner_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(model.inner_dim, model.config.tokenizer_vocab_size, bias=False)
        )

    model = torch.compile(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        model.add_adapter(lora_config)

    model.train()

    if args.image_to_text_only:
        frozen_keys = ["project_from_hidden", "up_block", "mlm_layer"]
        for n, p in model.named_parameters():
            if any([frozen_key in n for frozen_key in frozen_keys]):
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
    else:
        model.requires_grad_(True)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()   

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model_ in models:
                if isinstance(model_, type(accelerator.unwrap_model(model))):
                    if args.use_lora:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model_)
                    else:
                        model_.save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(model_, type(accelerator.unwrap_model(text_encoder))):
                    if args.text_encoder_use_lora:
                        text_encoder_lora_layers_to_save = get_peft_model_state_dict(model_)
                    else:
                        model_.save_pretrained(os.path.join(output_dir, "text_encoder"))
                else:
                    raise ValueError(f"unexpected save model: {model_.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if transformer_lora_layers_to_save is not None or text_encoder_lora_layers_to_save is not None:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=transformer_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )
                

    def load_model_hook(models, input_dir):
        transformer = None
        text_encoder_ = None

        # this part is added for keep consistency when add model.compile() in the model
        def adap_compile(ori_dict):#add '_orig_mod.' to each key
            new_dict = {}
            for k,v in ori_dict.items():
                new_dict['_orig_mod.' + k] = v
            return new_dict
            
        while len(models) > 0:
            model_ = models.pop()

            if isinstance(model_, type(accelerator.unwrap_model(model))):
                if args.use_lora:
                    transformer = model_
                else:
                    load_model = SymmetricTransformer2DModel.from_pretrained(os.path.join(input_dir, "transformer"), low_cpu_mem_usage=False, device_map=None)
                    model_.load_state_dict(adap_compile(load_model.state_dict()))
                    del load_model
            elif isinstance(model_, type(accelerator.unwrap_model(text_encoder))):
                if args.text_encoder_use_lora:
                    text_encoder_ = model_
                else:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(os.path.join(input_dir, "text_encoder"))
                        model_.load_state_dict(load_model.state_dict())
                        # print('finished loading text encoder!')
                    except:
                        print('Not found text-encoder model in current folder. So we download one text encoder from Internet.')
                        load_model = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                        model_.load_state_dict(load_model.state_dict())
                    del load_model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer is not None or text_encoder_ is not None:
            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
            )
            LoraLoaderMixin.load_lora_into_transformer(
                lora_state_dict, network_alphas=network_alphas, transformer=transformer
            )

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer_grouped_parameters = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "weight_decay": args.adam_weight_decay,
        }
    ]
    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if args.text_encoder_architecture == "t5_clip":
        tokenizer_for_dataset = [tokenizer, tokenizer_2]
    else:
        tokenizer_for_dataset = tokenizer

    if args.dataset_type == "ImageCaptionLargeDataset":
        dataset = ImageCaptionLargeDataset(
            root_dir=args.instance_data_dir,
            tokenizer=tokenizer_for_dataset,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture
        )
    elif args.dataset_type == "MultiSourceVLDataset":
        dataset = MultiSourceVLDataset(
            tokenizer=tokenizer_for_dataset,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            norm=False,
            llava_json=args.llava_json_path,
            llava_img_root=args.llava_image_root,
            mmmu_json=args.mmmu_json_path,
            mmmu_img_root=args.mmmu_image_root,
            vqa_ann_json=args.vqa_ann_json_path,
            vqa_img_root=args.vqa_image_root,
            coco_json=args.coco_json,
            coco_qa_json=args.coco_qa_json,
            coco_img_root=args.coco_img_root,
            gqa_json=args.gqa_json_root,
            gqa_img_root=args.gqa_image_root,
            mg_llava_json=args.mg_llava_json,
            mg_llava_root=args.mg_llava_root,
            caption_dir=args.caption_dir,
            pdd3_dir=args.instance_data_dir,
        )
    elif args.dataset_type == "DATA_TYPE":
        raise NotImplementedError("DATA_TYPE is not yet supported")
    else:
        assert False

    def collate_fn(samples):
        gen_images = [sample["gen_image"] for sample in samples]
        mmu_images = [sample["mmu_image"] for sample in samples]
        
        gen_micro_conds = [sample["gen_micro_conds"] for sample in samples]
        mmu_micro_conds = [sample["mmu_micro_conds"] for sample in samples]
        
        gen_images = torch.stack(gen_images, dim=0)
        mmu_images = torch.stack(mmu_images, dim=0)
        
        gen_micro_conds = torch.stack(gen_micro_conds, dim=0)
        mmu_micro_conds = torch.stack(mmu_micro_conds, dim=0)
        
        if isinstance(samples[0]["gen_input_ids"], list):
            gen_input_ids = [sample["gen_input_ids"][0] for sample in samples]
            gen_input_ids_2 = [sample["gen_input_ids"][1] for sample in samples]
            
            gen_input_ids = torch.cat(gen_input_ids, dim=0)
            gen_input_ids_2 = torch.cat(gen_input_ids_2, dim=0)
            gen_input_ids = [gen_input_ids, gen_input_ids_2]
        else:
            gen_input_ids = [sample["gen_input_ids"] for sample in samples]
            mmu_input_ids = [sample["mmu_input_ids"] for sample in samples]
            
            gen_input_ids = torch.cat(gen_input_ids, dim=0)
            mmu_input_ids = torch.cat(mmu_input_ids, dim=0)
        
        if samples[0].get("question_len", None) is not None:
            question_len = [sample["question_len"] for sample in samples]
            
            question_len = torch.cat(question_len, dim=0)   # [B, ]
        else:
            question_len = None
        
        ret = dict(
            gen_images=gen_images,
            mmu_images=mmu_images,
            gen_micro_conds=gen_micro_conds,
            mmu_micro_conds=mmu_micro_conds,
            gen_input_ids=gen_input_ids,
            mmu_input_ids=mmu_input_ids,
            question_len=question_len
        )
        
        return ret
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    train_dataloader.num_batches = len(train_dataloader)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    train_dataloader.num_batches = len(train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.text_encoder_architecture == "t5_clip":
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(device=accelerator.device, dtype=weight_dtype)
    else:
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
            
    vq_model.to(device=accelerator.device)

    with torch.no_grad():
        if args.text_encoder_architecture == "t5_clip":
            _input_ids_tmp_ = tokenize_prompt([tokenizer, tokenizer_2], "", args.text_encoder_architecture)
            _input_ids_tmp_[0] = _input_ids_tmp_[0].to(accelerator.device)
            _input_ids_tmp_[1] = _input_ids_tmp_[1].to(accelerator.device)
            empty_embeds, empty_clip_embeds = encode_prompt(
                [text_encoder, text_encoder_2], 
                _input_ids_tmp_, 
                args.text_encoder_architecture
            )
        else:
            _input_ids_tmp_ = tokenize_prompt(tokenizer, "", args.text_encoder_architecture)
            _input_ids_tmp_ = _input_ids_tmp_.to(accelerator.device)
            empty_embeds, empty_clip_embeds = encode_prompt(
                text_encoder,
                _input_ids_tmp_,
                args.text_encoder_architecture
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = { args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                gen_pixel_values = batch["gen_images"].to(accelerator.device, non_blocking=True)
                mmu_pixel_values = batch["mmu_images"].to(accelerator.device, non_blocking=True)
                
                gen_micro_conds = batch["gen_micro_conds"].to(accelerator.device, non_blocking=True)
                mmu_micro_conds = batch["mmu_micro_conds"].to(accelerator.device, non_blocking=True)
    
                # ====================== tokenize images ======================
                pixel_values = torch.cat([gen_pixel_values, mmu_pixel_values], dim=0)
                batch_size = pixel_values.shape[0]
                
                split_batch_size = args.split_vae_encode if args.split_vae_encode is not None else batch_size
                num_splits = math.ceil(batch_size / split_batch_size)
                image_tokens = []
                for i in range(num_splits):
                    start_idx = i * split_batch_size
                    end_idx = min((i + 1) * split_batch_size, batch_size)
                    image_tokens.append(
                        vq_model.quantize(
                            vq_model.encode(pixel_values[start_idx:end_idx]).latents
                        )[2][2].reshape(split_batch_size, -1)
                    )
                image_tokens = torch.cat(image_tokens, dim=0)
                gen_image_tokens, mmu_image_tokens = image_tokens.chunk(2, dim=0)
                # ====================== tokenize images ======================


                # ====================== encode clean text prompts ======================
                if args.text_encoder_architecture == "t5_clip":
                    gen_input_ids_clip = batch["gen_input_ids"][0].to(accelerator.device, non_blocking=True)
                    gen_input_ids_t5 = batch["gen_input_ids"][1].to(accelerator.device, non_blocking=True)
                    encoder_hidden_states, cond_embeds = encode_prompt(
                        [text_encoder, text_encoder_2], 
                        [gen_input_ids_clip, gen_input_ids_t5], 
                        args.text_encoder_architecture
                    )
                else:
                    gen_input_ids = batch["gen_input_ids"].to(accelerator.device, non_blocking=True)          
                    gen_encoder_hidden_states, gen_cond_embeds = encode_prompt(
                        text_encoder, 
                        gen_input_ids, 
                        args.text_encoder_architecture
                    )
                gen_encoder_hidden_states = gen_encoder_hidden_states.to(accelerator.device, dtype=accelerator.unwrap_model(model).dtype)
                gen_cond_embeds = gen_cond_embeds.to(accelerator.device, dtype=accelerator.unwrap_model(model).dtype)
                # ====================== encode clean text prompts ======================


                # ====================== image perturbation ======================
                half_batch_size, seq_len = gen_image_tokens.shape
                sigma = torch.rand(half_batch_size, device=gen_image_tokens.device)
                image_mask_prob = torch.cos(sigma * math.pi * 0.5)
                image_mask_prob = image_mask_prob.clip(args.min_masking_rate)
                image_timestep = image_mask_prob.clone().clamp(min=1e-3)

                num_token_masked = (seq_len * image_mask_prob).round().clamp(min=1)
                batch_randperm = torch.rand(half_batch_size, seq_len, device=gen_image_tokens.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)

                mask_id = accelerator.unwrap_model(model).config.vocab_size - 1
                gen_image_ids = torch.where(mask, mask_id, gen_image_tokens)
                image_labels = torch.where(mask, gen_image_tokens, -100)
                # ====================== image perturbation ======================


                # ====================== text perturbation ======================        
                if args.text_encoder_architecture == "t5_clip":
                    mmu_input_ids_clip = batch["mmu_input_ids"][0].to(accelerator.device, non_blocking=True)
                    mmu_input_ids_t5 = batch["mmu_input_ids"][1].to(accelerator.device, non_blocking=True)
                    half_batch_size, seq_len = mmu_input_ids_t5.shape
                    sigma = torch.rand(half_batch_size, device=mmu_image_tokens.device)
                    text_mask_prob = torch.cos(sigma * math.pi * 0.5)
                    text_mask_prob = text_mask_prob.clip(args.min_masking_rate)
                    text_timestep = text_mask_prob.clone().clamp(min=1e-3)

                    num_token_masked = (seq_len * text_mask_prob).round().clamp(min=1)
                    batch_randperm = torch.rand(half_batch_size, seq_len, device=mmu_image_tokens.device).argsort(dim=-1)
                    mask = batch_randperm < num_token_masked.unsqueeze(-1)

                    extra_id_0_token = "<extra_id_0>"
                    t5_mask_id = tokenizer_2.convert_tokens_to_ids(extra_id_0_token)
                    masked_prompt_input_ids_t5 = torch.where(mask, t5_mask_id, mmu_input_ids_t5)
                    text_labels = torch.where(mask, mmu_input_ids_t5, -100)
                    
                    # prepare input_ids for clip model
                    batch_prompt_2 = []
                    for i in range(masked_prompt_input_ids_t5.size(0)):
                        masked_prompt_input_id = masked_prompt_input_ids_t5[i].tolist()
                        prompt_2 = tokenizer_2.decode(masked_prompt_input_id, skip_special_tokens=True)
                        batch_prompt_2.append(prompt_2)
                    
                    masked_prompt_input_ids_clip = tokenizer(
                        batch_prompt_2,
                        truncation=True,
                        padding="max_length",
                        max_length=77,
                        return_tensors="pt"
                    ).input_ids
                    masked_prompt_input_ids_clip = masked_prompt_input_ids_clip.to(accelerator.device)
                else:
                    extra_id_0_token = "<extra_id_0>"
                    num_new_tokens = tokenizer.add_tokens(extra_id_0_token)
                    clip_mask_id = tokenizer.convert_tokens_to_ids(extra_id_0_token)
                    if num_new_tokens > 0:
                        text_encoder.resize_token_embeddings(len(tokenizer))
                        mask_token_embedding = text_encoder.get_input_embeddings().weight[clip_mask_id]
                        mask_token_embedding = mask_token_embedding.clone().detach().cpu().float()
                        if accelerator.is_main_process:
                            print("Saving masked token embedding...")
                            torch.save(mask_token_embedding, os.path.join(args.output_dir, "mask_token_embedding.pth"))
                            

                    mmu_input_ids = batch["mmu_input_ids"].to(accelerator.device, non_blocking=True) # [B, L]
                    question_len = batch["question_len"]    # [B, ]
                    if question_len is not None:
                        question_len = question_len.to(accelerator.device, non_blocking=True)
                                            
                    half_batch_size, seq_len = mmu_input_ids.shape
                    answer_len = seq_len - question_len
                    
                    sigma = torch.rand(half_batch_size, device=mmu_image_tokens.device)
                    text_mask_prob = torch.cos(sigma * math.pi * 0.5)
                    text_mask_prob = text_mask_prob.clip(args.min_masking_rate)
                    text_timestep = text_mask_prob.clone().clamp(min=1e-3)
                    
                    num_token_masked = ((seq_len - question_len) * text_mask_prob).round().clamp(min=1) # [B, ]
                    num_token_masked = torch.minimum(num_token_masked, answer_len)

                    seq_idx = torch.arange(seq_len, device=mmu_image_tokens.device).unsqueeze(0).repeat(half_batch_size, 1)
                    answer_region = seq_idx >= question_len.unsqueeze(1)
                    
                    rand_value = torch.rand(half_batch_size, seq_len, device=mmu_image_tokens.device)
                    rand_value = rand_value.masked_fill(~answer_region, float("inf"))
                    
                    order = rand_value.argsort(dim=-1)
                    order = order.argsort(dim=-1)
                    mask = order < num_token_masked.unsqueeze(-1)
                     
                    # mask = torch.zeros_like(mmu_input_ids)
                    # for b in range(half_batch_size):
                    #     ans_len = seq_len - question_len[b]
                    #     batch_randperm = torch.rand(1, ans_len, device=mmu_image_tokens.device).argsort(dim=-1)
                    #     mask[b, question_len[b]:] = batch_randperm < num_token_masked[b].unsqueeze(-1)
                            
                    mmu_input_ids_clip = torch.where(mask, clip_mask_id, mmu_input_ids)
                    text_labels = torch.where(mask, mmu_input_ids, -100)
                # ====================== text perturbation ======================


                # ====================== encode masked text prompts ======================
                if args.text_encoder_architecture == "t5_clip":
                    masked_encoder_hidden_states, masked_cond_embeds = encode_prompt(
                        [text_encoder, text_encoder_2], 
                        [masked_prompt_input_ids_clip, masked_prompt_input_ids_t5], 
                        args.text_encoder_architecture
                    )
                else:
                    mmu_encoder_hidden_states, mmu_cond_embeds = encode_prompt(
                        text_encoder, 
                        mmu_input_ids_clip, 
                        args.text_encoder_architecture
                    )
                mmu_encoder_hidden_states = mmu_encoder_hidden_states.to(accelerator.device, dtype=accelerator.unwrap_model(model).dtype)
                mmu_cond_embeds = mmu_cond_embeds.to(accelerator.device, dtype=accelerator.unwrap_model(model).dtype)
                # ====================== encode masked text prompts ======================
                
                
                # for CFG
                if args.cond_dropout_prob > 0.0:
                    assert encoder_hidden_states is not None

                    batch_size = encoder_hidden_states.shape[0]

                    mask = (
                        torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                        < args.cond_dropout_prob
                    )

                    empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                    encoder_hidden_states = torch.where(
                        (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                    )

                    empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                    cond_embeds = torch.where((cond_embeds * mask.squeeze(-1)).bool(), cond_embeds, empty_clip_embeds_)
                    
                vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
                resolution = args.resolution // vae_scale_factor
                gen_image_ids = gen_image_ids.reshape(half_batch_size, resolution, resolution)
                mmu_image_ids = mmu_image_tokens.reshape(half_batch_size, resolution, resolution)


            # Train Step
            with accelerator.accumulate(model):
                codebook_size = accelerator.unwrap_model(model).config.codebook_size                   
                if args.resolution == 1024: # only stage 3 and stage 4 do not apply 2*
                    img_ids = _prepare_latent_image_ids(
                        gen_image_ids.shape[0], 
                        gen_image_ids.shape[-2],
                        gen_image_ids.shape[-1],
                        gen_image_ids.device,
                        gen_image_ids.dtype
                    )
                else:
                    img_ids = _prepare_latent_image_ids(
                        gen_image_ids.shape[0],
                        gen_image_ids.shape[-2],
                        gen_image_ids.shape[-1],
                        gen_image_ids.device,
                        gen_image_ids.dtype
                    )

                txt_ids = torch.zeros(gen_encoder_hidden_states.shape[1], 3).to(device=gen_image_ids.device, dtype=gen_image_ids.dtype)
                
                image_logits = model(
                    hidden_states=gen_image_ids, # should be (batch size, channel, height, width)
                    encoder_hidden_states=gen_encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                    micro_conds=gen_micro_conds,
                    pooled_projections=gen_cond_embeds, # should be (batch_size, projection_dim)
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    timestep=image_mask_prob * 1000,
                )[0]
                image_logits = image_logits.reshape(half_batch_size, codebook_size, -1)
                image_logits = image_logits.permute(0, 2, 1)
                image_logits = image_logits.reshape(-1, codebook_size)

                image_loss = F.cross_entropy(
                    image_logits,
                    image_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                image_loss = image_loss.reshape(half_batch_size, -1).mean(-1)
                image_loss = image_loss / image_timestep
                image_loss = image_loss.mean()

                text_logits = model(
                    hidden_states=mmu_image_ids, # should be (batch size, channel, height, width)
                    encoder_hidden_states=mmu_encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                    micro_conds=mmu_micro_conds,
                    pooled_projections=mmu_cond_embeds, # should be (batch_size, projection_dim)
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    timestep=text_timestep * 1000,
                )[1]
                text_logits = text_logits.reshape(-1, accelerator.unwrap_model(model).config.tokenizer_vocab_size)

                text_loss = F.cross_entropy(
                    text_logits,
                    text_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                text_loss = text_loss.reshape(half_batch_size, -1).mean(-1)
                text_loss = text_loss / text_timestep
                text_loss = text_loss.mean()

                loss = image_loss + args.text_loss_weight * text_loss
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_masking_rate = accelerator.gather(text_mask_prob.repeat(args.train_batch_size)).mean()

                accelerator.backward(loss)
                
                # Temporarily add this to identify unused parameters
                # for name, param in accelerator.unwrap_model(model).named_parameters():
                #     if param.grad is None:
                #         print(f"Unused parameter: {name}")

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if (global_step + 1) % args.logging_steps == 0:
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1, logger)

                if (global_step + 1) % args.validation_steps == 0 and accelerator.is_main_process:

                    with torch.no_grad():
                        logger.info("Generating images...")

                        model.eval()

                        scheduler = Scheduler.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="scheduler",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        
                        pipe = UnifiedPipeline(
                            transformer=accelerator.unwrap_model(model),
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            vqvae=vq_model,
                            scheduler=scheduler,
                            tokenizer_2=tokenizer_2,
                            text_encoder_2=text_encoder_2,
                        )

                        if not args.image_to_text_only:
                            output = pipe(
                                prompt=args.validation_prompts,
                                height=args.resolution,
                                width=args.resolution,
                                guidance_scale=9,
                                num_inference_steps=64,
                            )
                            pil_images = output.images

                            result=[]
                            for img in pil_images:
                                if not isinstance(img, torch.Tensor):
                                    img = transforms.ToTensor()(img)
                                result.append(img.unsqueeze(0))
                            result = torch.cat(result,dim=0)
                            result = make_grid(result, nrow=3)
                            save_image(result,os.path.join(args.output_dir, str(global_step)+'_text2image_1024_CFG-9.png'))
                        
                            output_data = {
                                "step": global_step,
                                "prompts": args.validation_prompts,
                                "images": [f"{global_step}_text2image_1024_CFG-9_{i}.png" for i in range(len(pil_images))]
                            }

                            with open(os.path.join(args.output_dir, f"text2image_{global_step}.json"), "w") as f:
                                json.dump(output_data, f, indent=2)

                        image = load_images_to_tensor(args.validation_images, target_size=(args.resolution, args.resolution))        
                        output = pipe(
                            prompt=args.validation_vqa_prompts,
                            height=args.resolution, 
                            width=args.resolution, 
                            guidance_scale=9,
                            image=image,
                            num_inference_steps=64
                        )
                        prompts = output.prompts
                        
                        output_data = {
                            "step": global_step,
                            "prompts": prompts,
                        }

                        with open(os.path.join(args.output_dir, f"image2text_{global_step}.json"), "w") as f:
                            json.dump(output_data, f, indent=2)

                        model.train()

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step, logger)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())