"""
Feature encoding utilities for dataset preprocessing.

This module provides functions for encoding:
- Video to VAE latent representation
- First frame to latent (for i2v conditioning)
- Text prompts (via TextEncoder)
- First frame visual features (via VisionEncoder)
- ByT5 glyph text
"""

import re
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torchvision import transforms


def encode_video_to_latent(
    vae: nn.Module,
    video_frames: torch.Tensor,
    target_height: int,
    target_width: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode video to VAE latent representation.

    Processing pipeline:
    1. Resize + CenterCrop to target resolution
    2. Normalize to [-1, 1]
    3. VAE encode
    4. Multiply by scaling_factor

    Args:
        vae: VAE model
        video_frames: [T, H, W, C] uint8 tensor
        target_height: Target height (must be divisible by 16)
        target_width: Target width (must be divisible by 16)
        device: Device for computation

    Returns:
        Latent tensor [1, C_latent, T_latent, H_latent, W_latent] float32

    Note:
        VAE temporal structure: frame 0 → latent 0, frames 1..4 → latent 1, etc.
        Required frame count: 1 + 4 * (L - 1) where L is number of latent frames
    """
    H, W = video_frames.shape[1], video_frames.shape[2]

    # Resize + CenterCrop to target resolution
    if H != target_height or W != target_width:
        scale_factor = max(target_width / W, target_height / H)
        resize_h = int(round(H * scale_factor))
        resize_w = int(round(W * scale_factor))

        # [T, H, W, C] → [T, C, H, W] for interpolate
        frames = video_frames.permute(0, 3, 1, 2).float()  # [T, C, H, W]
        frames = torch.nn.functional.interpolate(
            frames, size=(resize_h, resize_w), mode="bilinear", align_corners=False
        )

        # Center crop
        crop_top = (resize_h - target_height) // 2
        crop_left = (resize_w - target_width) // 2
        frames = frames[
            :,
            :,
            crop_top : crop_top + target_height,
            crop_left : crop_left + target_width,
        ]

        # Normalize to [-1, 1]
        video = frames / 127.5 - 1.0  # [T, C, H, W]
    else:
        video = video_frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # [T, C, H, W]

    # Add batch dimension and reshape to [B, C, T, H, W]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

    vae_dtype = next(vae.parameters()).dtype
    video = video.to(device, dtype=vae_dtype)

    with torch.no_grad():
        latent = vae.encode(video).latent_dist.sample()
        # latent: [B, C_latent, T_latent, H_latent, W_latent]
        latent = latent * vae.config.scaling_factor

    return latent.cpu().float()


def encode_first_frame_to_latent(
    vae: nn.Module,
    first_frame: torch.Tensor,
    target_height: int,
    target_width: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode first frame to VAE latent (for image_cond in i2v).

    Processing pipeline:
    1. Resize + CenterCrop to target resolution
    2. Normalize with [0.5]
    3. VAE encode (using mode, not sample)
    4. Multiply by scaling_factor

    Args:
        vae: VAE model
        first_frame: [H, W, C] uint8 tensor
        target_height: Target height
        target_width: Target width
        device: Device for computation

    Returns:
        Image condition latent [1, C_latent, 1, H_latent, W_latent] float32
    """
    # Convert to PIL Image
    frame_np = first_frame.numpy()
    pil_image = Image.fromarray(frame_np)

    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resize_width = int(round(original_width * scale_factor))
    resize_height = int(round(original_height * scale_factor))

    # Transform consistent with inference pipeline
    ref_image_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_height, resize_width),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    ref_images_pixel_values = ref_image_transform(pil_image)
    # [C, H, W] → [1, C, 1, H, W]
    ref_images_pixel_values = (
        ref_images_pixel_values.unsqueeze(0).unsqueeze(2).to(device)
    )

    vae_dtype = next(vae.parameters()).dtype
    ref_images_pixel_values = ref_images_pixel_values.to(dtype=vae_dtype)

    with torch.no_grad():
        # Use mode() instead of sample(), consistent with inference
        cond_latents = vae.encode(ref_images_pixel_values).latent_dist.mode()
        cond_latents = cond_latents * vae.config.scaling_factor

    return cond_latents.cpu().float()  # [1, C_latent, 1, H_latent, W_latent]


def encode_prompt(
    prompt: str,
    text_encoders: Dict,
    device: str = "cuda",
    max_length: int = 1000,
) -> Dict[str, torch.Tensor]:
    """
    Encode text prompt using TextEncoder.

    Args:
        prompt: Text prompt string
        text_encoders: Dictionary with 'text_encoder' key
        device: Device for computation
        max_length: Maximum token length

    Returns:
        Dictionary with:
            - prompt_embeds: [1, seq_len, dim]
            - prompt_mask: [1, seq_len]
    """
    text_encoder = text_encoders["text_encoder"]

    with torch.no_grad():
        # Use TextEncoder API
        text_inputs = text_encoder.text2tokens(
            prompt, data_type="video", max_length=max_length
        )

        prompt_outputs = text_encoder.encode(
            text_inputs, data_type="video", device=device
        )

        prompt_embeds = prompt_outputs.hidden_state
        prompt_mask = prompt_outputs.attention_mask

        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device)

    return {
        "prompt_embeds": prompt_embeds.cpu(),
        "prompt_mask": prompt_mask.cpu(),
    }


def encode_first_frame(
    first_frame: torch.Tensor,
    vision_encoder_dict: Dict,
    target_height: int,
    target_width: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Encode first frame visual features (for i2v vision_states).

    Processing pipeline:
    1. Resize and center crop to target resolution
    2. VisionEncoder.encode_images(numpy_array)

    Args:
        first_frame: [H, W, C] uint8 tensor
        vision_encoder_dict: Dictionary with 'vision_encoder' key
        target_height: Target height
        target_width: Target width
        device: Device for computation

    Returns:
        Dictionary with:
            - vision_states: [1, seq_len, dim]
    """
    vision_encoder = vision_encoder_dict["vision_encoder"]

    # Resize and center crop to target resolution
    frame_np = first_frame.numpy()  # [H, W, C] uint8
    pil_image = Image.fromarray(frame_np)
    original_width, original_height = pil_image.size

    scale_factor = max(target_width / original_width, target_height / original_height)
    resize_width = int(round(original_width * scale_factor))
    resize_height = int(round(original_height * scale_factor))

    # Resize
    pil_image = pil_image.resize((resize_width, resize_height), Image.LANCZOS)
    # Center crop
    left = (resize_width - target_width) // 2
    top = (resize_height - target_height) // 2
    pil_image = pil_image.crop((left, top, left + target_width, top + target_height))

    input_image_np = np.array(pil_image)

    with torch.no_grad():
        # Use VisionEncoder encode_images method
        vision_outputs = vision_encoder.encode_images(input_image_np)
        vision_states = vision_outputs.last_hidden_state  # [1, seq_len, dim]

    return {
        "vision_states": vision_states.cpu(),
    }


def encode_byt5_prompt(
    prompt: str,
    byt5_dict: Dict,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Encode byT5 glyph text prompt.

    Extracts text inside quotes (single or double) and formats it
    using the MultilingualPromptFormat for glyph rendering.

    Args:
        prompt: Text prompt (may contain quoted text for glyphs)
        byt5_dict: Dictionary with byt5 model components, or None
        device: Device for computation

    Returns:
        Dictionary with:
            - byt5_text_states: [1, seq_len, dim]
            - byt5_text_mask: [1, seq_len]

    Note:
        If byt5_dict is None or no quoted text found, returns zero tensors
    """
    if byt5_dict is None:
        # If byT5 is not loaded, return zero tensors
        logger.warning("byT5 not loaded, using zero tensors")
        return {
            "byt5_text_states": torch.zeros(1, 256, 1472),
            "byt5_text_mask": torch.zeros(1, 256, dtype=torch.int64),
        }

    byt5_model = byt5_dict["byt5_model"]
    byt5_tokenizer = byt5_dict["byt5_tokenizer"]
    byt5_max_length = byt5_dict["byt5_max_length"]

    # Extract text inside quotes (if any)
    pattern = r'"(.*?)"|"(.*?)"'
    matches = re.findall(pattern, prompt)
    glyph_texts = [match[0] or match[1] for match in matches]

    if len(glyph_texts) == 0:
        # No quoted text, return zero tensors
        return {
            "byt5_text_states": torch.zeros(1, byt5_max_length, 1472).to(device),
            "byt5_text_mask": torch.zeros(1, byt5_max_length, dtype=torch.int64).to(
                device
            ),
        }

    # Format text
    prompt_format = byt5_dict["prompt_format"]
    text_styles = [
        {"color": None, "font-family": None} for _ in range(len(glyph_texts))
    ]
    formatted_text = prompt_format.format_prompt(glyph_texts, text_styles)

    # Tokenize
    byt5_text_inputs = byt5_tokenizer(
        formatted_text,
        padding="max_length",
        max_length=byt5_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    text_ids = byt5_text_inputs.input_ids.to(device)
    text_mask = byt5_text_inputs.attention_mask.to(device)

    with torch.no_grad():
        byt5_outputs = byt5_model(text_ids, attention_mask=text_mask.float())
        byt5_embeddings = byt5_outputs[0]

    return {
        "byt5_text_states": byt5_embeddings.cpu(),
        "byt5_text_mask": text_mask.cpu(),
    }