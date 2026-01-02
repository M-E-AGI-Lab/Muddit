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
import pandas as pd


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
        for root, _, filenames in os.walk(root_dir):
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
        allava_json: str = None, allava_img_root: str = None,
        vqav2_json: str = None, vqav2_img_root: str = None,
        gqa_json: str = None, gqa_img_root: str = None,
        cider_root: str = None,
        vqav2_root: str = None,
        mme_root: str = None,
        mmb_root: str = None,
        mmmu_root: str = None,
        img_root: str = None, caption_dir: str = None,
        geneval_root: str = None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.arch = text_encoder_architecture
        self.norm = norm

        self.gen_samples = []      # [(img_path, prompt), ...]
        self.mmu_samples = []      # [(img_path, question, answer), ...]

        if llava_json:
            self._load_llava_data(llava_json, llava_img_root)
        if allava_json:
            self._load_instruction_tuning_data(allava_json, allava_img_root)
        if vqav2_json:
            self._load_instruction_tuning_data(vqav2_json, vqav2_img_root)
        if vqav2_root:
            self._load_vqav2_data(vqav2_root)
        if cider_root:
            self._load_cider_data(cider_root)
        if gqa_json:
            self._load_instruction_tuning_data(gqa_json, gqa_img_root)
        if mmmu_root:
            self._load_mm_data(mmmu_root)
        if mmb_root:
            self._load_mm_data(mmb_root)
        if mme_root:
            self._load_mme_data(mme_root)
        if caption_dir:
            self._load_caption(caption_dir)
        if img_root:
            self._load_image(img_root)
        if geneval_root:
            self._load_image(geneval_root)

        self.len_mmu = len(self.mmu_samples)
        self.len_gen = len(self.gen_samples)
        
        print(f"[INFO] Loaded {self.len_mmu} MMU samples and {self.len_gen} generation samples.")

    # ------------------------------------------------------------------ #
    #                          dataset parsers                           #
    # ------------------------------------------------------------------ #
    def _load_llava_data(self, json_path, img_root):
        before_len = len(self.mmu_samples)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for ex in data:
            image = ex.get("image", None)
            if image is not None:
                img_file = os.path.join(img_root, ex["image"])
                
                human_msgs = [m["value"] for m in ex["conversations"] if m["from"] == "human"]
                gpt_msgs = [m["value"] for m in ex["conversations"] if m["from"] == "gpt"]
                
                for human_msg, gpt_msg in zip(human_msgs, gpt_msgs):
                    human_msg = human_msg.replace("<image>", "").strip()
                    gpt_msg = gpt_msg.strip()
                    self.mmu_samples.append((img_file, human_msg, gpt_msg)) 

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {json_path}.")


    def _load_instruction_tuning_data(self, json_path, img_root):
        if img_root is None:
            img_root = os.path.dirname(json_path)
        
        before_len = len(self.mmu_samples)
        
        data = []
        cnt = 0
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        for ex in data:
            image = ex.get("image", None)
            if image is not None:
                img_file = os.path.join(img_root, ex["image"])
                
                if os.path.exists(img_file):
                    human_msgs = [m["value"] for m in ex["conversations"] if m["from"] == "human"]
                    gpt_msgs = [m["value"] for m in ex["conversations"] if m["from"] == "gpt"]
                    
                    for human_msg, gpt_msg in zip(human_msgs, gpt_msgs):
                        human_msg = human_msg.replace("<image>", "").strip()
                        gpt_msg = gpt_msg.strip()
                        self.mmu_samples.append((img_file, human_msg, gpt_msg))
                        cnt += 1

            if "gqa" in json_path:
                if cnt > 300000:
                    break

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {json_path}.")
    

    def _load_cider_data(self, root_dir):
        before_len = len(self.mmu_samples)
        json_filepath = os.path.join(root_dir, "captions_val2014.json")
        image_root = os.path.join(root_dir, "val2014")
        if os.path.exists(json_filepath):
            with open(json_filepath, "r") as file:
                data = json.load(file)

            for item in data["annotations"]:
                # image_id is 395, then the image path is image_root/COCO_val2014_000000000395.jpg
                img_path = os.path.join(image_root, f"COCO_val2014_{item['image_id']:012d}.jpg")
                if os.path.exists(img_path):
                    question = "Please describe this image concisely."
                    caption = item["caption"]
                    self.mmu_samples.append((img_path, question, caption))

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {root_dir}.")


    def _load_vqav2_data(self, root_dir):
        before_len = len(self.mmu_samples)

        json_filepath = os.path.join(root_dir, "validation_annotations.json")
        if os.path.exists(json_filepath):
            with open(json_filepath, "r") as file:
                data = json.load(file)
            for item in data:
                img_path = os.path.join(root_dir, item["image"])
                if os.path.exists(img_path):
                    question = item["question"]
                    answer = item["answer"][0]["answer"]
                    self.mmu_samples.append((img_path, question, answer))

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {root_dir}.")


    def _load_mme_data(self, root_dir):
        before_len = len(self.mmu_samples)
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png")):
                    try:
                        base = os.path.splitext(f)[0]
                        txt_path = os.path.join(root, base + ".txt")
                        txt_path = txt_path.replace("images", "questions_answers_YN")
                        if os.path.exists(txt_path):
                            with open(txt_path, "r") as file:
                                caption = file.read().strip()
                            case1, case2 = caption.split("\n")
                            
                            q1, a1 = case1.split(" Please answer yes or no.")
                            q1 = q1.strip() + " Please answer yes or no."
                            a1 = a1.strip()
                            q2, a2 = case2.split(" Please answer yes or no.")
                            q2 = q2.strip() + " Please answer yes or no."
                            a2 = a2.strip()
                            self.mmu_samples.append((os.path.join(root, f), q1, a1))
                            self.mmu_samples.append((os.path.join(root, f), q2, a2))
                            
                    except Exception as e:
                        print(f"[Warning] Error at {os.path.join(root, f)}: {e}")
                        continue

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {root_dir}.")


    def _load_caption(self, root_dir):
        before_len = len(self.mmu_samples)
        
        jsonl_file = os.path.join(root_dir, "images.jsonl")
        if os.path.exists(jsonl_file):
            with open(jsonl_file, "r") as file:
                for line in file:
                    item = json.loads(line)
                    img_path = os.path.join(root_dir, item["image"])
                    if os.path.exists(img_path):
                        q = "Please describe this image in detail."
                        caption = item["caption"]
                        self.mmu_samples.append((img_path, q, caption))
                        
        else:
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png")):
                        base = os.path.splitext(f)[0]
                        txt_path = os.path.join(root, base + ".txt")
                        if os.path.exists(txt_path):
                            with open(txt_path, "r") as file:
                                caption = file.read().strip()
                            q = "Please describe this image in detail."
                            self.mmu_samples.append((os.path.join(root, f), q, caption))

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {root_dir}.")
        

    def _load_image(self, root_dir):
        before_len = len(self.gen_samples)
        
        jsonl_file = os.path.join(root_dir, "images.jsonl")
        if os.path.exists(jsonl_file):
            with open(jsonl_file, "r") as file:
                for line in file:
                    item = json.loads(line)
                    img_path = os.path.join(root_dir, item["image"])
                    caption = item["caption"]
                    self.gen_samples.append((img_path, caption))
                        
        else:
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png")):
                        base = os.path.splitext(f)[0]
                        txt_path = os.path.join(root, base + ".txt")
                        if os.path.exists(txt_path):
                            with open(txt_path, "r") as file:
                                caption = file.read().strip()
                            self.gen_samples.append((os.path.join(root, f), caption))

        after_len = len(self.gen_samples)
        print(f"[INFO] Loaded {after_len - before_len} from {root_dir}.")


    def _load_mm_data(self, root_dir):
        """
        Data example:
        
        Question: Identify the question that Madelyn and Tucker's experiment can best answer.

        A. Does Madelyn's snowboard slide down a hill in less time when it has a thin layer of wax or a thick layer of wax?
        B. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?

        Answer: B
        
        Args:
            root_dir (str): Path to the MMBench directory containing parquet files
        """
        before_len = len(self.mmu_samples)
        if not os.path.exists(root_dir):
            print(f"[Warning] MMBench directory not found: {root_dir}")
            return
        
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png")):
                    try:
                        base = os.path.splitext(f)[0]
                        txt_path = os.path.join(root, base + ".txt")
                        if os.path.exists(txt_path):
                            with open(txt_path, "r") as file:
                                caption = file.read().strip()

                            question = caption.split("Answer: ")[0].replace("Question: ", "").replace("\n", "").strip()
                            answer = caption.split("Answer: ")[1].strip()
                            
                            self.mmu_samples.append((os.path.join(root, f), question, answer))
                    except Exception as e:
                        print(f"[Warning] Error at {os.path.join(root, f)}: {e}")
                        continue

        after_len = len(self.mmu_samples)
        print(f"[INFO] Loaded {after_len - before_len} samples from {root_dir}.")

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