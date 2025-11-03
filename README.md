# 🌌 Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model

[![arXiv](https://img.shields.io/badge/arXiv-2505.23606-b31b1b.svg)](https://arxiv.org/abs/2505.23606)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model_Muddit-yellow)](https://huggingface.co/MeissonFlow/Muddit)
[![Demo](https://img.shields.io/badge/Live-Demo_Muddit-blue?logo=huggingface)](https://huggingface.co/spaces/MeissonFlow/muddit)

[Chinese Media Report](https://www.techwalker.com/2025/0603/3167162.shtml)


![Tracing the Evolution of Unified Generation Foundation Models](./Evolution.png)

## 📝 Meissonic Updates and Family Papers

- [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) [CVPR 2022]
- [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/abs/2301.00704) [ICML 2023]
- [🌟][Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis](https://arxiv.org/abs/2410.08261) [ICLR 2025]
- [Bag of Design Choices for Inference of High-Resolution Masked Generative Transformer](https://arxiv.org/abs/2411.10781)
- [Di[𝙼]O: Distilling Masked Diffusion Models into One-step Generator](https://arxiv.org/abs/2503.15457) [ICCV 2025]
- [🌟][Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606)
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
- More papers are coming soon!
See [MeissonFlow Research](https://huggingface.co/MeissonFlow) (Organization Card) for more about our vision.

## 🚀 Introduction

Welcome to the official repository of **Muddit** — a next-generation foundation model in the Meissonic family, built upon discrete diffusion for unified and efficient multimodal generation.

Unlike traditional autoregressive methods, **Muddit** leverages discrete diffusion (a.k.a. MaskGIT-style masking) as its core mechanism — enabling fast, parallel decoding across modalities.

While most unified models are still rooted in language priors, **Muddit** is developed from a visual-first perspective for scalable and flexible generation.

Muddit (512) and Muddit Plus (1024) aim to handle diverse tasks across modalities -- such as text generation, image generation, and vision-language reasoning -- within a single architecture and decoding paradigm.

---

## 💡 Inference Usage

### Gradio Web UI

Please refer to https://huggingface.co/spaces/MeissonFlow/muddit/blob/main/app.py

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


---

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
