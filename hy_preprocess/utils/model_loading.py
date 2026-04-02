"""
Model loading utilities for HunyuanVideo preprocessing.

This module provides unified model loading functions for:
- VAE (Variational Autoencoder)
- Text Encoder (LLM-based)
- Vision Encoder (SigLIP)
- ByT5 Encoder (for glyph text)
"""

import os
from typing import Dict, Optional

import torch
from loguru import logger


def load_vae_model(model_path: str, device: str = "cuda", dtype=None):
    """
    Load HunyuanVideo VAE model.

    Args:
        model_path: Path to HunyuanVideo model root directory
        device: Device to load model on (default: "cuda")
        dtype: Data type for model parameters. If None, automatically selects
               based on GPU memory (float16 for <23GB, float32 otherwise)

    Returns:
        VAE model in eval mode with gradients disabled
    """
    from hyvideo.commons import get_gpu_memory
    from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache

    logger.info(f"Loading VAE from {model_path}")
    vae_path = os.path.join(model_path, "vae")

    if dtype is None:
        memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        dtype = torch.float16 if memory_limitation < 23 * GB else torch.float32

    vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
        vae_path,
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    logger.info(f"VAE loaded successfully with dtype={dtype}")
    return vae


def load_text_encoder(model_path: str, device: str = "cuda"):
    """
    Load LLAMA text encoder for prompt encoding.

    Args:
        model_path: Path to HunyuanVideo model root directory
        device: Device to load model on

    Returns:
        Dictionary with 'text_encoder' key

    Raises:
        FileNotFoundError: If text encoder path doesn't exist
    """
    from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder

    logger.info("Loading text encoders...")

    text_encoder_path = os.path.join(model_path, "text_encoder", "llm")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(
            f"{text_encoder_path} not found. Please check your model path."
        )

    text_encoder = TextEncoder(
        text_encoder_type="llm",
        tokenizer_type="llm",
        text_encoder_path=text_encoder_path,
        max_length=1000,
        text_encoder_precision="fp16",
        prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
        prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        logger=logger,
        device=device,
    )

    logger.info("Text encoder loaded successfully")
    return {"text_encoder": text_encoder}


def load_vision_encoder(model_path: str, device: str = "cuda"):
    """
    Load SigLIP vision encoder for image conditioning (i2v).

    Args:
        model_path: Path to HunyuanVideo model root directory
        device: Device to load model on

    Returns:
        Dictionary with 'vision_encoder' key

    Raises:
        FileNotFoundError: If vision encoder path doesn't exist
    """
    from hyvideo.models.vision_encoder import VisionEncoder

    logger.info("Loading vision encoder...")

    vision_encoder_path = os.path.join(model_path, "vision_encoder", "siglip")
    if not os.path.exists(vision_encoder_path):
        raise FileNotFoundError(
            f"{vision_encoder_path} not found. Please check your model path."
        )

    vision_encoder = VisionEncoder(
        vision_encoder_type="siglip",
        vision_encoder_precision="fp16",
        vision_encoder_path=vision_encoder_path,
        processor_type=None,
        processor_path=None,
        output_key=None,
        logger=logger,
        device=device,
    )

    logger.info("Vision encoder loaded successfully")
    return {"vision_encoder": vision_encoder}


def load_byt5_encoder(
    model_path: str, device: str = "cuda", byt5_max_length: int = 256
) -> Optional[Dict]:
    """
    Load byT5 encoder for glyph text encoding.

    Args:
        model_path: Path to HunyuanVideo model root directory
        device: Device to load model on
        byt5_max_length: Maximum sequence length for byT5

    Returns:
        Dictionary with byt5 model components, or None if glyph checkpoint not found
    """
    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
    from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat

    logger.info("Loading byT5 encoder...")

    glyph_root = os.path.join(model_path, "text_encoder", "Glyph-SDXL-v2")
    if not os.path.exists(glyph_root):
        logger.warning(
            f"Glyph checkpoint not found from '{glyph_root}'. Skipping byT5 loading."
        )
        return None

    byT5_google_path = os.path.join(model_path, "text_encoder", "byt5-small")
    if not os.path.exists(byT5_google_path):
        logger.warning(
            f"ByT5 google path not found from: {byT5_google_path}. "
            f"Using 'google/byt5-small' from HuggingFace."
        )
        byT5_google_path = "google/byt5-small"

    multilingual_prompt_format_color_path = os.path.join(
        glyph_root, "assets/color_idx.json"
    )
    multilingual_prompt_format_font_path = os.path.join(
        glyph_root, "assets/multilingual_10-lang_idx.json"
    )

    byt5_args = dict(
        byT5_google_path=byT5_google_path,
        byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
        multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
        multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
        byt5_max_length=byt5_max_length,
    )

    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
    prompt_format = MultilingualPromptFormat(
        font_path=multilingual_prompt_format_font_path,
        color_path=multilingual_prompt_format_color_path,
    )

    logger.info("byT5 encoder loaded successfully")

    return {
        "byt5_model": byt5_kwargs["byt5_model"],
        "byt5_tokenizer": byt5_kwargs["byt5_tokenizer"],
        "byt5_max_length": byt5_kwargs["byt5_max_length"],
        "prompt_format": prompt_format,
    }
