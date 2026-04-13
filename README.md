# Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model

[![arXiv](https://img.shields.io/badge/arXiv-2505.23606-b31b1b.svg)](https://arxiv.org/abs/2505.23606)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model_Muddit-yellow)](https://huggingface.co/MeissonFlow/Muddit)
[![Demo](https://img.shields.io/badge/Live-Demo_Muddit-blue?logo=huggingface)](https://huggingface.co/spaces/MeissonFlow/muddit)


## 🚀 Introduction

Muddit is the 2nd generation [Meissonic](https://github.com/viiika/Meissonic). It is built upon discrete diffusion for unified and efficient multimodal generation.

Unlike traditional autoregressive methods, **Muddit** leverages discrete diffusion with absorbing kernel (a.k.a. MaskGIT-style masking) as its core mechanism — enabling fast, parallel decoding across modalities.

While most unified models are still rooted in language priors, **Muddit** is developed from a visual-first perspective for scalable and flexible generation. This aligns Jim Fan's [The Second Pre-training Paradigm](https://x.com/DrJimFan/status/2018754323141054786).

Muddit (512) and Muddit Plus (1024) aim to handle diverse tasks across modalities -- such as text generation, image generation, and vision-language reasoning -- within a single architecture and decoding paradigm.


![Tracing the Evolution of Unified Generation Foundation Models](./Evolution.png)
*Note: Data as of July 2025.

---

## Highlights

- **Unified discrete diffusion** for both text and image generation under a shared MM-DiT backbone.
- **Built on Meissonic visual priors**, instead of training a unified model from scratch.
- **One model, three tasks**: text-to-image, image captioning, and VQA.
- **1B parameters** with competitive performance against much larger autoregressive multimodal models.
- **Fast parallel decoding** with strong real-world throughput on both image and text generation.

## Supported Tasks

- Text-to-Image: Generate high-quality images from text prompts.

- Image-to-Text: Generate descriptive captions from input images.

- Visual Question Answering: Answer questions conditioned on both image and text.

## Benchmark Snapshot

### Main results

| Model | Params | GenEval | MS-COCO CIDEr | VQAv2 | MME | GQA | MMMU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Muddit (512) | 1B | 0.61 | 59.9 | 68.2 | 1107.4 | 57.5 | 27.6 |
| Muddit (1024) | 1B | - | 60.1 | 70.2 | 1139.2 | 57.8 | 28.7 |

### Efficiency

| Model | Resolution | Steps | Text-to-Image (img/s) | Image-to-Text (token/s) |
|---|---:|---:|---:|---:|
| UniDisc | 512 | 32 | 0.89 | 79.36 |
| D-DiT | 512 | 28 | 0.62 | 26.89 |
| Muddit | 512 | 32 | 1.00 | 99.98 |

Muddit matches the best reported text-to-image throughput in this comparison while achieving the fastest image-to-text decoding among the listed non-autoregressive baselines.



## 💡 Inference

### Installation

```bash
git clone https://github.com/M-E-AGI-Lab/Muddit.git
cd Muddit
pip install -r requirements.txt
```

### Gradio Web UI

Please refer to https://huggingface.co/spaces/MeissonFlow/muddit/blob/main/app.py

### Text-to-Image
```bash
bash inference_t2i.sh 
```

### Image-to-Text/VQA
```bash
bash inference_i2t.sh
```

## 🎓 Training

To train Muddit, follow these steps:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your own dataset and dataset class following the format in [dataset_utils.py](./train/dataset_utils.py) and [train_meissonic.py](./train/train_unified.py)
   - Modify [train.sh](./train/train_unified.sh) with your dataset path

4. Start training:
   ```bash
   bash train/train_unified.sh
   ```

Note: For custom datasets, you'll likely need to implement your own dataset class.

## Technical Overview

Muddit uses a shared MM-DiT generator for both text and image tokens. During training, one modality is randomly masked and reconstructed conditioned on the other. During inference, Muddit starts from fully masked tokens and iteratively denoises them in parallel.

This gives a genuinely unified framework where the same diffusion process supports:

- text → image
- image → text
- image + question → answer

---
## 📝 Meissonic Updates and Family Papers

- [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) [CVPR 2022]
- [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/abs/2301.00704) [ICML 2023]
- [🌟][Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis](https://arxiv.org/abs/2410.08261) [ICLR 2025]
- [Bag of Design Choices for Inference of High-Resolution Masked Generative Transformer](https://arxiv.org/abs/2411.10781)
- [Di[𝙼]O: Distilling Masked Diffusion Models into One-step Generator](https://arxiv.org/abs/2503.15457) [ICCV 2025]
- [🌟][Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606) [ICLR 2026]
- [DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](https://arxiv.org/pdf/2507.04947) [ICCV 2025]
- [MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control](https://arxiv.org/abs/2508.10684)
- [Lavida-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2509.19244)
- [🌟][Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding](https://arxiv.org/abs/2510.06308)
- [Token Painter: Training-Free Text-Guided Image Inpainting via Mask Autoregressive Models](https://arxiv.org/abs/2509.23919)
- [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171)
- [OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows](https://arxiv.org/abs/2510.03506)
- [Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces](https://arxiv.org/abs/2506.07903) [ICML 2025]
- [Towards Better & Faster Autoregressive Image Generation: From the Perspective of Entropy](https://arxiv.org/abs/2510.09012) [NeurIPS 2025]
- [🌟][From Masks to Worlds: A Hitchhiker's Guide to World Models](https://arxiv.org/abs/2510.20668)
- [Soft-Di[M]O: Improving One-Step Discrete Image Generation with Soft Embeddings](https://arxiv.org/abs/2509.22925)
- [Accelerating Inference of Masked Image Generators via Reinforcement Learning](https://arxiv.org/abs/2512.01094)
- [Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models](https://arxiv.org/abs/2512.14008)
- [Reinforcement Learning Meets Masked Generative Models: Mask-GRPO for Text-to-Image Generation](https://arxiv.org/abs/2510.13418)
- [MaskFocus: Focusing Policy Optimization on Critical Steps for Masked Image Generation](https://arxiv.org/abs/2512.18766)
- [Co-GRPO: Co-Optimized Group Relative Policy Optimization for Masked Diffusion Model](https://arxiv.org/abs/2512.22288)
- [Synergistic Multimodal Diffusion Transformer: Unifying and Enhancing Multimodal Generation via Adaptive Discrete Diffusion](https://www.preprints.org/frontend/manuscript/0aaaa5c92ad47499c4bfdcc229baf1d1/download_pub#page=2.28)
- [🌟][Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](https://arxiv.org/abs/2602.01842)

- More papers are coming soon!
See [MeissonFlow Research](https://huggingface.co/MeissonFlow) (Organization Card) for more about our vision.


## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{shi2025muddit,
  title={Muddit: Liberating generation beyond text-to-image with a unified discrete diffusion model},
  author={Shi, Qingyu and Bai, Jinbin and Zhao, Zhuoran and Chai, Wenhao and Yu, Kaidong and Wu, Jianzong and Song, Shuangyong and Tong, Yunhai and Li, Xiangtai and Li, Xuelong and others},
  journal={arXiv preprint arXiv:2505.23606},
  year={2025}
}
```

---

<p align="center">
  <a href="https://star-history.com/#M-E-AGI-Lab/Muddit&Date">
    <img src="https://api.star-history.com/svg?repos=M-E-AGI-Lab/Muddit&type=Date" alt="Star History Chart">
  </a>
</p>

<p align="center">
  Made with ❤️ by the MeissonFlow Research
  
  See [MeissonFlow Research](https://huggingface.co/MeissonFlow) (Organization Card) for more about our vision.
</p>
