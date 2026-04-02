"""
Utility modules for dataset preprocessing.

This package contains shared functions for:
- Model loading (VAE, Text, Vision, ByT5 encoders)
- Feature encoding (video, image, text)
- Video processing (loading, resampling)
- Pose/camera data processing
"""

from .model_loading import (
    load_vae_model,
    load_text_encoder,
    load_vision_encoder,
    load_byt5_encoder,
)
from .encoding import (
    encode_video_to_latent,
    encode_first_frame_to_latent,
    encode_prompt,
    encode_first_frame,
    encode_byt5_prompt,
)
from .video_utils import (
    load_video_segment,
    load_image_sequence,
    resample_video_frames,
)
from .pose_utils import (
    convert_npz_to_pose_and_actions,
    convert_json_to_pose_and_actions,
    convert_gamefactory_actions_to_pose_and_actions,
)

__all__ = [
    "load_vae_model",
    "load_text_encoder",
    "load_vision_encoder",
    "load_byt5_encoder",
    "encode_video_to_latent",
    "encode_first_frame_to_latent",
    "encode_prompt",
    "encode_first_frame",
    "encode_byt5_prompt",
    "load_video_segment",
    "load_image_sequence",
    "resample_video_frames",
    "convert_npz_to_pose_and_actions",
    "convert_json_to_pose_and_actions",
    "convert_gamefactory_actions_to_pose_and_actions",
]
