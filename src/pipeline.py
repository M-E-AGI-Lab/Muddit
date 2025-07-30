import os
import sys
from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import torch
import PIL
import numpy as np
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5Tokenizer, T5EncoderModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import VQModel
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.scheduler import Scheduler
from src.transformer import SymmetricTransformer2DModel


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> image = pipe(prompt).images[0]
        ```
"""


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def dedup_consecutive_words(text: str) -> str:
    # Weired problem
    text = text.replace("is such is", "").replace("such is", "").replace("such as", "").replace("such", "")
    words = text.split()
    if not words:
        return text

    out = [words[0]]
    for w in words[1:]:
        if w != out[-1]:
            out.append(w)
    return " ".join(out)

def keep_upto_last_period(text: str) -> str:
    # Try the Chinese full stop first
    idx = text.rfind("。")
    # Fallback to the ASCII period
    if idx == -1:
        idx = text.rfind(".")
    # If still not found, return original text
    if idx == -1:
        return text
    # Keep everything up to (and including) the last period
    return text[:idx + 1]

@dataclass
class UnifiedPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    prompts: List[str]


class UnifiedPipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    tokenizer_2: T5Tokenizer
    text_encoder: CLIPTextModelWithProjection
    text_encoder_2: T5EncoderModel
    transformer: SymmetricTransformer2DModel
    scheduler: Scheduler
    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: SymmetricTransformer2DModel,
        scheduler: Scheduler,
        tokenizer_2: T5Tokenizer = None,
        text_encoder_2: T5EncoderModel = None,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        image: Optional[Union[torch.Tensor, PIL.Image.Image]] = None,
        num_inference_steps: int = 48,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.IntTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        output_type = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        mask_token_embedding: Optional[str] = None,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                gneration. If not provided, the starting latents will be completely masked.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.

        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """
        if (prompt_embeds is not None and encoder_hidden_states is None) or (
            prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

        if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
            negative_prompt_embeds is None and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )
        
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(self._execution_device)

        text2image = image is None
        image2text = image is not None

        if image2text:
            if self.text_encoder_2 is not None:
                prompt = "<extra_id_0>" * 256
                prompt = [prompt] * len(image)
                
                t5_mask_id = self.tokenizer_2.convert_tokens_to_ids("<extra_id_0>")
                self.scheduler.config.mask_token_id = t5_mask_id
            else:
                mask_token = "<mask>"
                self.tokenizer.add_tokens(mask_token, special_tokens=False)
                clip_mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                
                if mask_token_embedding is not None:
                    if mask_token_embedding.endswith(".pth"):
                        mask_token_embedding = torch.load(mask_token_embedding)
                    else:
                        mask_token_embedding = os.path.dirname(mask_token_embedding)
                        mask_token_embedding_path = os.path.join(mask_token_embedding, "mask_token_embedding.pth")
                        assert os.path.exists(mask_token_embedding_path), f"{mask_token_embedding_path} doesn't exists!"
                        mask_token_embedding = torch.load(mask_token_embedding_path)
                        
                    mask_token_embedding = mask_token_embedding.to(self._execution_device, dtype=self.text_encoder.dtype)
                    self.text_encoder.get_input_embeddings().weight.data[clip_mask_id].copy_(mask_token_embedding)
                
                self.scheduler.config.mask_token_id = clip_mask_id
                
                input_ids = torch.ones(
                    size=(len(image), self.tokenizer.model_max_length),
                    dtype=torch.int64,
                    device=self._execution_device
                )
                input_ids = input_ids * clip_mask_id
                
                question_len = []
                if prompt is None:
                    question_len = [0] * len(image)
                elif isinstance(prompt, str):
                    question_ids = torch.LongTensor([self.tokenizer.encode(prompt)])
                    question_ids = question_ids.repeat(len(image), 1)
                    
                    q_len = len(question_ids[0]) - 1   # remove <eos> token
                    question_len = [q_len] * len(image)
                    
                    input_ids[:, :q_len] = question_ids[:, :-1]
                else: 
                    assert isinstance(prompt, list), f"prompt must be None or str or list!"
                    assert len(prompt) == len(image), f"VQA require equal num of images and prompts!"
                    for i, p in enumerate(prompt):
                        question_ids = torch.LongTensor([self.tokenizer.encode(p)])
                        
                        q_len = len(question_ids[0]) - 1
                        question_len.append(q_len)
                        
                        input_ids[i, :q_len] = question_ids[0, :-1]
        else:
            self.scheduler.config.mask_token_id = self.transformer.config.vocab_size - 1

        if isinstance(prompt, str):
            prompt = [prompt]

        if image is not None:
            batch_size = len(image)
        else:
            batch_size = len(prompt)

        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor

        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        text_encoder_type = "t5_clip" if self.text_encoder_2 is not None else "open_clip"
        if prompt_embeds is None:
            if text_encoder_type == "t5_clip":
                if text2image:
                    input_ids_clip = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=77,
                    ).input_ids.to(self._execution_device)
                    outputs = self.text_encoder(input_ids_clip, return_dict=True, output_hidden_states=True)
                    prompt_embeds = outputs.text_embeds
                    
                    input_ids_t5 = self.tokenizer_2(
                        prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=256,
                    ).input_ids.to(self._execution_device)
                outputs_2 = self.text_encoder_2(input_ids_t5, return_dict=True, output_hidden_states=True)
                encoder_hidden_states = outputs_2.last_hidden_state
            else:
                if text2image:
                    input_ids = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=77,
                    ).input_ids.to(self._execution_device)
                    
                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                prompt_embeds = outputs.text_embeds
                encoder_hidden_states = outputs.hidden_states[-2]

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        if guidance_scale > 1.0 and text2image:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * len(prompt)

                if text_encoder_type == "t5_clip":                    
                    input_ids = self.tokenizer(
                        negative_prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=77,
                    ).input_ids.to(self._execution_device)
                    outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                    negative_prompt_embeds = outputs.text_embeds
                    
                    input_ids_2 = self.tokenizer_2(
                        negative_prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=256,
                    ).input_ids.to(self._execution_device)
                    outputs_2 = self.text_encoder_2(input_ids_2, return_dict=True, output_hidden_states=True)
                    negative_encoder_hidden_states = outputs_2.last_hidden_state
                
                else:
                    input_ids = self.tokenizer(
                        negative_prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=77,
                    ).input_ids.to(self._execution_device)

                    outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                
                    negative_prompt_embeds = outputs.text_embeds
                    negative_encoder_hidden_states = outputs.hidden_states[-2]
                          
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        # Note that the micro conditionings _do_ flip the order of width, height for the original size
        # and the crop coordinates. This is how it was done in the original code base
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 and text2image else batch_size, -1)

        shape = (batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None and text2image:
            latents = torch.full(
                shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=self._execution_device
            )
        elif image2text:
            if text_encoder_type == "t5_clip":
                latents = input_ids_t5 # [b, l]
            else:
                latents = input_ids

        model_input = None

        step_by_step = []

        self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(self.scheduler.timesteps):
                if guidance_scale > 1.0 and text2image:
                    model_input = torch.cat([latents] * 2)
                    encoder_hidden_states = encoder_hidden_states
                elif image2text:
                    if model_input is None:
                        model_input = self.vqvae.quantize(
                            self.vqvae.encode(image.to(self._execution_device, dtype=self.vqvae.dtype)).latents
                        )[2][2].reshape(batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    
                    if text_encoder_type == "t5_clip":
                        outputs_t5 = self.text_encoder_2(latents, return_dict=True)
                        encoder_hidden_states = outputs_t5.last_hidden_state
                        
                        batch_prompt = []
                        for i in range(latents.size(0)):
                            masked_prompt_input_id = latents[i].tolist()
                            prompt = self.tokenizer_2.decode(masked_prompt_input_id, skip_special_tokens=True)
                            batch_prompt.append(prompt)
                        
                        masked_prompt_input_ids_clip = self.tokenizer(
                            batch_prompt,
                            truncation=True,
                            padding="max_length",
                            max_length=77,
                            return_tensors="pt"
                        ).input_ids
                        masked_prompt_input_ids_clip = masked_prompt_input_ids_clip.to(self._execution_device)
                        outputs_clip = self.text_encoder(input_ids=masked_prompt_input_ids_clip, return_dict=True)
                        prompt_embeds = outputs_clip.text_embeds
                        
                    else:
                        outputs = self.text_encoder(latents, return_dict=True, output_hidden_states=True)
                        prompt_embeds = outputs.text_embeds
                        encoder_hidden_states = outputs.hidden_states[-2]
                else:
                    model_input = latents
                    encoder_hidden_states = encoder_hidden_states
                    
                if height == 1024: #args.resolution == 1024:
                    img_ids = _prepare_latent_image_ids(
                        model_input.shape[0], 
                        model_input.shape[-2],
                        model_input.shape[-1],
                        model_input.device,
                        model_input.dtype
                    )
                else:
                    img_ids = _prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[-2],
                        model_input.shape[-1],
                        model_input.device,
                        model_input.dtype
                    )
                txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype
                )
                
                # timestep_ = int(timestep / num_inference_steps * 1000)
                model_output, encoder_hidden_states_tmp = self.transformer(
                    hidden_states=model_input,
                    micro_conds=micro_conds,
                    pooled_projections=prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    timestep=torch.tensor([timestep], device=model_input.device, dtype=torch.long),
                )

                if image2text:
                    encoder_hidden_states = encoder_hidden_states_tmp.clone()

                if guidance_scale > 1.0 and text2image:
                    uncond_logits, cond_logits = model_output.chunk(2)
                    to_scheduler = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                elif image2text:
                    to_scheduler = encoder_hidden_states
                else:
                    to_scheduler = model_output

                latents = self.scheduler.step(
                    model_output=to_scheduler,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

                # this line will print the intermediate results of the image-to-text generation
                # step_by_step.append(self.tokenizer.decode(latents[0].tolist(), skip_special_tokens=True))
                
                # this line will print the intermediate results of the text-to-image generation
                # output = self.vqvae.decode(
                #     latents,
                #     force_not_quantize=True,
                #     shape=(
                #         batch_size,
                #         height // self.vae_scale_factor,
                #         width // self.vae_scale_factor,
                #         self.vqvae.config.latent_channels,
                #     ),
                # ).sample.clip(0, 1)
                # output = self.image_processor.postprocess(output, output_type)    # output is a list of PIL.Image, you need to save it.

                if i == len(self.scheduler.timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents)

        # with open("step_by_step.txt", "w") as file:
        #     for prompt in step_by_step:
        #         file.write(prompt + "\n")

        if guidance_scale > 1.0 and text2image:
            decoded_input_ids = encoder_hidden_states[encoder_hidden_states.shape[0] // 2:].argmax(-1)
        else:
            decoded_input_ids = encoder_hidden_states.argmax(-1)
            
        prompts = []
        for i, prompt in enumerate(decoded_input_ids):
            if image2text:
                q_len = question_len[i]
                prompt = self.tokenizer.decode(prompt.tolist()[q_len:], skip_special_tokens=True)        
                prompts.append(keep_upto_last_period(dedup_consecutive_words(prompt)))
            else:
                prompts.append("Placeholder")
            
        if output_type == "latent":
            output = latents
        else:
            needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast

            if needs_upcasting:
                self.vqvae.float()

            if text2image:
                to_vqvae = latents
            else:
                to_vqvae = model_input
                
            output = self.vqvae.decode(
                to_vqvae,
                force_not_quantize=True,
                shape=(
                    batch_size,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                    self.vqvae.config.latent_channels,
                ),
            ).sample.clip(0, 1)
            output = self.image_processor.postprocess(output, output_type)

            if needs_upcasting:
                self.vqvae.half()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (output,)

        return UnifiedPipelineOutput(images=output, prompts=prompts)