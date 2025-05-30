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

import os
import shutil
from pathlib import Path, PosixPath

import torch
from PIL import Image
from torchvision import transforms


def save_checkpoint(args, accelerator, global_step, logger):
    output_dir = args.output_dir

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def load_images_to_tensor(path, target_size=(1024, 1024)):
    """
    Args:
        folder_path
        target_size: (height, width)
    
    Return:
        torch.Tensor: [B, 3, H, W] in [0, 1]
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    if isinstance(path, list):
        image_files = path
    elif isinstance(path, str) and os.path.isdir(path):
        image_files = [f for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    elif isinstance(path, str):
        image_files = [path]
    else:
        raise ValueError(f"Unsupported folder_path type: {type(path)}")
    
    if not image_files:
        raise ValueError(f"No valid images found in {path}")
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    tensors = []
    for img_file in image_files:
        try:
            if isinstance(path, str) and os.path.isdir(path):
                img_path = os.path.join(path, img_file)
            else:
                img_path = img_file
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            tensors.append(tensor)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    if not tensors:
        raise ValueError("No images could be loaded")
    
    batch_tensor = torch.stack(tensors)
    
    assert batch_tensor.shape[1:] == (3, *target_size), \
        f"Output shape is {batch_tensor.shape}, expected (B, 3, {target_size[0]}, {target_size[1]})"
    
    return batch_tensor