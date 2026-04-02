"""
Video processing utilities for dataset preprocessing.

This module provides functions for:
- Loading video segments from MP4 files
- Loading image sequences from directories
- Frame resampling and temporal alignment
"""

import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image


def load_video_segment(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    num_frames: Optional[int] = None,
) -> torch.Tensor:
    """
    Load video segment from MP4 file.

    Args:
        video_path: Path to MP4 video file
        start_frame: Start frame index (default: 0)
        end_frame: End frame index (default: last frame)
        num_frames: Alternative: load first N frames (mutually exclusive with end_frame)

    Returns:
        Tensor of shape [T, H, W, C] with uint8 dtype

    Note:
        If num_frames is provided, it overrides end_frame.
        Useful for datasets where video length should match camera pose count.
    """
    # Lazy import to avoid dependency issues
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)

    # Determine frame range
    if num_frames is not None:
        # Load first num_frames (for matching camera poses)
        actual_frames = min(total_frames, num_frames)
        frame_indices = list(range(actual_frames))
    else:
        # Load specified range
        start_frame = 0 if start_frame is None else max(0, start_frame)
        end_frame = total_frames - 1 if end_frame is None else min(total_frames - 1, end_frame)
        frame_indices = list(range(start_frame, end_frame + 1))

    frames = vr.get_batch(frame_indices).asnumpy()
    del vr

    return torch.from_numpy(frames.astype(np.uint8))


def load_image_sequence(folder_path: str) -> torch.Tensor:
    """
    Load image sequence from directory (frame_*.png).

    Args:
        folder_path: Path to directory containing frame_*.png files

    Returns:
        Tensor of shape [T, H, W, C] with uint8 dtype

    Raises:
        FileNotFoundError: If no frame_*.png files found

    Note:
        Files are sorted naturally (frame_2 before frame_10)
    """
    search_pattern = os.path.join(folder_path, "frame_*.png")
    img_paths = glob.glob(search_pattern)

    if not img_paths:
        raise FileNotFoundError(f"No frame_*.png found in {folder_path}")

    # Natural sort: ensure frame_2 comes before frame_10
    img_paths.sort(
        key=lambda x: int(re.search(r"frame_(\d+)", os.path.basename(x)).group(1))
    )

    frames = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        frames.append(np.array(img))

    # Return shape [T, H, W, C], dtype uint8
    return torch.from_numpy(np.stack(frames).astype(np.uint8))


def resample_video_frames(
    video_frames: torch.Tensor,
    target_num_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Resample video frame sequence to target number of frames.

    Uses uniform temporal sampling with index interpolation:
    - Compute source indices as i * (T-1) / (target_num_frames-1)
    - Round to nearest integer
    - Sample corresponding frames

    Args:
        video_frames: [T, H, W, C] uint8 tensor
        target_num_frames: Target frame count; None means no resampling

    Returns:
        Tuple of:
            - resampled_frames: [T', H, W, C] uint8 tensor
            - source_indices: List of original frame indices (0-based ints)

    Raises:
        ValueError: If target_num_frames <= 0

    Examples:
        >>> frames = torch.randn(100, 480, 832, 3)  # 100 frames
        >>> resampled, indices = resample_video_frames(frames, 129)
        >>> print(resampled.shape)  # [129, 480, 832, 3]
        >>> print(len(indices))  # 129
    """
    T = video_frames.shape[0]

    if target_num_frames is None or target_num_frames == T:
        return video_frames, list(range(T))

    if target_num_frames <= 0:
        raise ValueError(
            f"target_num_frames must be a positive integer, got {target_num_frames}"
        )

    if target_num_frames == 1:
        return video_frames[:1], [0]

    # Uniformly distributed source indices (float), convert to int then sample frames
    source_indices: List[int] = [
        int(round(i * (T - 1) / (target_num_frames - 1)))
        for i in range(target_num_frames)
    ]

    # Ensure indices are within bounds
    source_indices = [min(i, T - 1) for i in source_indices]
    frames_out = video_frames[source_indices]

    return frames_out, source_indices
