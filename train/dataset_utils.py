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
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from PIL import Image
import io
import json
import numpy as np
import pyarrow.parquet as pq
import random
import bisect
import pyarrow.fs as fs


@torch.no_grad()
def tokenize_prompt(
    tokenizer, 
    prompt, 
    text_encoder_architecture='open_clip',
    padding='max_length',
    max_length=77,
    max_length_t5=256,
):
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids
        return input_ids
    elif text_encoder_architecture == 't5_clip':
        input_ids = []
        input_ids.append(tokenizer[0](
            prompt,
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids)
        input_ids.append(tokenizer[1](
            prompt,
            truncation=True,
            padding=padding,
            max_length=max_length_t5,
            return_tensors="pt",
        ).input_ids)
        return input_ids
    elif text_encoder_architecture == "gemma":
        input = tokenizer(
            prompt,
            truncation=True,
            padding=padding,
            padding_side="right",
            max_length=max_length,
            return_tensors="pt",
        )
        return input  
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")


def encode_prompt(
    text_encoder, 
    input_ids, 
    text_encoder_architecture='open_clip'
):
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        outputs = text_encoder(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        cond_embeds = outputs[0]
        return encoder_hidden_states, cond_embeds
    elif text_encoder_architecture == 't5_clip':
        outputs_clip = text_encoder[0](
            input_ids=input_ids[0], 
            return_dict=True, 
            output_hidden_states=True
        )
        outputs_t5 = text_encoder[1](
            input_ids=input_ids[1], 
            return_dict=True, 
            output_hidden_states=True
        )
        encoder_hidden_states = outputs_t5.last_hidden_state
        cond_embeds = outputs_clip.text_embeds
        return encoder_hidden_states, cond_embeds
    elif text_encoder_architecture == "gemma":
        outputs = text_encoder(**input_ids.to(text_encoder.device))
        encoder_hidden_states = outputs.last_hidden_states
        cond_embeds = encoder_hidden_states.mean(dim=-2)
        return encoder_hidden_states, cond_embeds
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")


def process_image(image, size, Norm=False, hps_score=6.0): 
    image = exif_transpose(image)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    orig_height = image.height
    orig_width = image.width

    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(size, size))
    image = transforms.functional.crop(image, c_top, c_left, size, size)
    image = transforms.ToTensor()(image)

    if Norm:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)

    micro_conds = torch.tensor(
        [orig_width, orig_height, c_top, c_left, hps_score],
    )

    return {"image": image, "micro_conds": micro_conds}    
    

class ImageCaptionLargeDataset(Dataset):
    def __init__(
        self, 
        root_dir,
        tokenizer,
        size,
        text_encoder_architecture="CLIP",
        norm=False
    ):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.size = size
        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm
        
        self.data_list = []        
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    base_name = os.path.splitext(filename)[0]
                    txt_file = os.path.join(root, base_name + ".txt")
                    if os.path.exists(txt_file):
                        self.data_list.append((root, base_name + ".txt", filename))
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        try:
            sub_dir, txtfilename, imgfilename = self.data_list[idx]
            img_path = os.path.join(sub_dir, imgfilename)
            caption_path = os.path.join(sub_dir, txtfilename)
            
            image = Image.open(img_path).convert("RGB")
            ret = process_image(image, self.size, self.norm)
            
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            
            ret["prompt_input_ids"] = tokenize_prompt(
                self.tokenizer, caption, self.text_encoder_architecture
            )
            
            return ret
        
        except Exception as e:
            print("===========================================")
            print(f"[Warning] Error at index {idx}: {img_path}")
            print("===========================================")
            if idx + 1 < len(self.data_list):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(len(self.data_list) - 1)


class MultiSourceVLDataset(Dataset):
    """
    A unified dataloader for
      • LLaVA-Instruct-150K
      • MMMU (multiple-choice QA)
      • VQAv2
      • Local caption files under `pdd3/`
    """

    def __init__(
        self,
        tokenizer,
        size: int,
        text_encoder_architecture: str = "CLIP",
        norm: bool = False,
        # ----- paths -----
        llava_json: str = None, llava_img_root: str = None,
        mmmu_json: str = None,  mmmu_img_root: str = None,
        vqa_ann_json: str = None, vqa_img_root: str = None,
        gqa_json: str = None, gqa_img_root: str = None,
        coco_json: str = None, coco_img_root: str = None,
        coco_qa_json: str = None,
        mg_llava_json: str = None, mg_llava_root: str = None,
        pdd3_dir: str = None, caption_dir: str = None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.arch = text_encoder_architecture
        self.norm = norm

        self.gen_samples = []      # [(img_path, prompt), ...]
        self.mmu_samples = []      # [(img_path, question, answer), ...]

        if llava_json:
            self._load_llava(llava_json, llava_img_root)
        if mmmu_json:
            self._load_mmmu(mmmu_json, mmmu_img_root)
        if vqa_ann_json:
            self._load_vqav2(vqa_ann_json, vqa_img_root)
        if mg_llava_json:
            self._load_mg_llava(mg_llava_json, mg_llava_root)
        if caption_dir:
            self._load_caption(caption_dir)
        if pdd3_dir:
            self._load_pdd3(pdd3_dir)

        self.len_mmu = len(self.mmu_samples)
        self.len_gen = len(self.gen_samples)

    # ------------------------------------------------------------------ #
    #                          dataset parsers                            #
    # ------------------------------------------------------------------ #
    def _load_llava(self, json_path, img_root):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for ex in data:
            img_file = os.path.join(img_root, ex["image"])

            human_msg = next(m["value"] for m in ex["conversations"] if m["from"] == "human")
            gpt_msg   = next(m["value"] for m in ex["conversations"] if m["from"] == "gpt")

            self.mmu_samples.append((img_file, human_msg.strip(), gpt_msg.strip())) 

    def _load_mmmu(self, json_path, img_root):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for ex in data:
            img_file = os.path.join(img_root, ex["image"])
            choices  = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"])])
            
            question = f"{ex['question'].strip()}\n{choices}"
            answer = f"{ex['answer']}"
            
            self.mmu_samples.append((img_file, question, answer))

    def _load_vqav2(self, ann_json, img_root):
        with open(ann_json, "r") as file:
            annos = json.load(file)

        for ann in annos:
            q = ann["question"]
            answer = ann["answer"]
            img_path = ann["image"]
            img_file = os.path.join(
                img_root,
                img_path   # if val, modify to val2014
            )

            self.mmu_samples.append((img_file, q, answer))

    def _load_mg_llava(self, json_path, img_root):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for ex in data:
            image = ex.get("image", None)
            if image is not None:
                img_file = os.path.join(img_root, ex["image"])
                if os.path.exists(img_file):
                    human_msg = next(m["value"] for m in ex["conversations"] if m["from"] == "human")
                    gpt_msg   = next(m["value"] for m in ex["conversations"] if m["from"] == "gpt")

                    self.mmu_samples.append((img_file, human_msg.strip(), gpt_msg.strip())) 

    def _load_caption(self, root_dir):
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png")):
                    base = os.path.splitext(f)[0]
                    txt_path = os.path.join(root, base + ".txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r") as file:
                            caption = file.read().strip()
                        q = "Please describe this image."
                        self.mmu_samples.append((os.path.join(root, f), q, caption))

    def _load_pdd3(self, root_dir):
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png")):
                    base = os.path.splitext(f)[0]
                    txt_path = os.path.join(root, base + ".txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r") as file:
                            caption = file.read().strip()
                        self.gen_samples.append((os.path.join(root, f), caption))

    # ------------------------------------------------------------------ #
    #                       PyTorch Dataset API                          #
    # ------------------------------------------------------------------ #
    def __len__(self):
        return max(self.len_gen, self.len_mmu)

    def __getitem__(self, idx):
        get_mmu_data = False
        get_gen_data = False
        
        while not get_mmu_data:
            try:
                mmu_img_path, question, answer = self.mmu_samples[idx]
                get_mmu_data = True
            except:
                idx = random.randint(0, self.len_mmu - 1)
            
        while not get_gen_data:
            try:
                gen_img_path, prompt = self.gen_samples[idx]
                get_gen_data = True
            except:
                idx = random.randint(0, self.len_gen - 1)

        try:
            # ---- image ----
            mmu_image = Image.open(mmu_img_path).convert("RGB")
            mmu_ret = process_image(mmu_image, self.size, self.norm)
            
            gen_image = Image.open(gen_img_path).convert("RGB")
            gen_ret = process_image(gen_image, self.size, self.norm)

            ret = dict(
                gen_image=gen_ret["image"],
                gen_micro_conds=gen_ret["micro_conds"],
                mmu_image=mmu_ret["image"],
                mmu_micro_conds=mmu_ret["micro_conds"]
            )

            # ---- text ----
            question = question.replace("<image>", "").replace("\n", "")
            question_ids = tokenize_prompt(
                self.tokenizer, 
                question, 
                self.arch,
                padding=False,
            )
            question_ids = question_ids[:, :-1]
            q_len = len(question_ids[0])
            if answer:
                full_prompt = question + " " + answer
            else:
                full_prompt = question
            mmu_input_ids = tokenize_prompt(self.tokenizer, full_prompt, self.arch)
            
            gen_input_ids = tokenize_prompt(self.tokenizer, prompt, self.arch)

            ret.update({
                "gen_input_ids": gen_input_ids,
                "mmu_input_ids": mmu_input_ids,
                "question_len": torch.LongTensor([q_len])
            })
            return ret
        except:
            print("================================================================")
            print(f"There is something wrong with {mmu_img_path} or {gen_img_path}.")
            print("================================================================")
            if idx < self.len_gen - 1 or idx < self.len_mmu - 1:
                return self.__getitem__(idx + 1)
            else:
                idx = random.randint(0, self.len_gen - 1)
                return self.__getitem__(idx)