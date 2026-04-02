"""
Camera pose and action processing utilities.

This module provides functions for converting different pose formats:
- NPZ files (extrinsic/intrinsic matrices)
- JSON files (frame-based w2c/intrinsic)
- GameFactory metadata (action-based poses)
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def convert_npz_to_pose_and_actions(
    npz_path: str,
    source_frame_indices: List[int],
) -> Tuple[Dict, Dict]:
    """
    Read camera matrices from NPZ file directly.

    Assumes extrinsic is c2w (Camera-to-World), converts to w2c as pipeline requires.

    Args:
        npz_path: Path to NPZ file with 'intrinsic' and 'extrinsic' arrays
        source_frame_indices: List of frame indices to extract

    Returns:
        Tuple of:
            - pose_dict: {"0": {"w2c": ..., "intrinsic": ...}, ...}
            - action_dict: {"0": {"move_action": "", "view_action": ""}, ...}
    """
    camera_data = np.load(npz_path)

    # Intrinsic is typically (3, 3) fixed for the video
    intrinsic_base = camera_data["intrinsic"]
    intrinsic_list = intrinsic_base.tolist()

    # Extrinsic is typically (N, 4, 4)
    extrinsic_array = camera_data["extrinsic"]
    total_poses = extrinsic_array.shape[0]

    pose_dict: Dict = {}
    action_dict: Dict = {}

    for out_idx, frame_offset in enumerate(source_frame_indices):
        frame_offset = int(frame_offset)
        # Prevent index out of bounds
        frame_offset = min(max(0, frame_offset), total_poses - 1)

        c2w = extrinsic_array[frame_offset]

        # Invert c2w to get w2c for pipeline
        try:
            w2c = np.linalg.inv(c2w)
        except np.linalg.LinAlgError:
            w2c = np.eye(4)  # Fallback identity if matrix is somehow singular

        pose_dict[str(out_idx)] = {
            "w2c": w2c.tolist(),
            "intrinsic": intrinsic_list,
        }
        # Provide dummy action strings for generic video
        action_dict[str(out_idx)] = {
            "move_action": "",
            "view_action": "",
        }

    return pose_dict, action_dict


def convert_json_to_pose_and_actions(
    json_path: str,
    source_frame_indices: List[int],
) -> Tuple[Dict, Dict]:
    """
    Extract w2c and intrinsic from JSON with 'frame_0', 'frame_1' keys.

    Original JSON provides w2c directly, no inversion needed.

    Args:
        json_path: Path to JSON file with frame-based pose entries
        source_frame_indices: List of frame indices to extract

    Returns:
        Tuple of pose_dict and action_dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        camera_data = json.load(f)

    pose_dict: Dict = {}
    action_dict: Dict = {}

    for out_idx, frame_offset in enumerate(source_frame_indices):
        key = f"frame_{int(frame_offset)}"

        if key not in camera_data:
            logger.warning(
                f"Key {key} not found in {json_path}. Falling back to identity matrix."
            )
            w2c = np.eye(4).tolist()
            intrinsic = [
                [0.5, 0, 0.5],
                [0, 0.888, 0.5],
                [0, 0, 1],
            ]  # Dummy fallback
        else:
            w2c = camera_data[key]["w2c"]
            intrinsic = camera_data[key]["intrinsic"]

        # Convert to dense keys "0", "1", "2"...
        pose_dict[str(out_idx)] = {
            "w2c": w2c,
            "intrinsic": intrinsic,
        }
        # Fill empty actions
        action_dict[str(out_idx)] = {
            "move_action": "",
            "view_action": "",
        }

    return pose_dict, action_dict


def _pose_from_action_data(action_data: Dict) -> np.ndarray:
    """
    Compute W2C matrix (4x4 float32) from single-frame GameFactory action_data.

    Args:
        action_data: Dictionary with 'pos', 'pre_pitch', 'pre_yaw'

    Returns:
        W2C transformation matrix (4x4)
    """
    pos = action_data.get("pos", [0.0, 0.0, 0.0])
    pitch = action_data.get("pre_pitch", 0.0)
    yaw = action_data.get("pre_yaw", 0.0)

    pitch_rad, yaw_rad = np.deg2rad(pitch), np.deg2rad(yaw)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)

    R = np.array(
        [
            [cos_yaw, -sin_yaw * cos_pitch, sin_yaw * sin_pitch],
            [sin_yaw, cos_yaw * cos_pitch, -cos_yaw * sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    return np.linalg.inv(c2w)


def _action_from_action_data(action_data: Dict) -> Tuple[str, str]:
    """
    Extract (move_action, view_action) strings from single-frame GameFactory action_data.

    Args:
        action_data: Dictionary with game action fields

    Returns:
        Tuple of move_action string and view_action string
    """
    ws = action_data.get("ws", 1)  # 0=S, 1=none, 2=W
    ad = action_data.get("ad", 0)  # 0=none, 1=A, 2=D

    move_action = ""
    if ws == 2:
        move_action += "W"
    elif ws == 0:
        move_action += "S"
    if ad == 2:
        move_action += "D"
    elif ad == 1:
        move_action += "A"

    pitch_delta = action_data.get("pitch_delta", 0.0)
    yaw_delta = action_data.get("yaw_delta", 0.0)

    view_action = ""
    if abs(yaw_delta) > 0.1:
        if yaw_delta > 0:
            view_action = "LL"  # turn left
        else:
            view_action = "LR"  # turn right
    elif abs(pitch_delta) > 0.1:
        if pitch_delta > 0:
            view_action = "LD"  # look down
        else:
            view_action = "LU"  # look up

    return move_action, view_action


def convert_gamefactory_actions_to_pose_and_actions(
    metadata: Dict,
    start_frame: int,
    end_frame: int,
    target_height: int = 480,
    target_width: int = 832,
    source_frame_indices: Optional[List[int]] = None,
) -> Tuple[Dict, Dict]:
    """
    Convert GameFactory metadata to pose and action format for training.

    Important: CameraJsonWMemDataset uses pose_keys[4*(i-1)+4] for positional indexing,
    so pose_json and action_json must contain entries for **every output video frame**,
    with keys "0", "1", "2", ..., "num_output_frames-1".

    Args:
        metadata: GameFactory metadata dict
        start_frame: Start frame of original video (for indexing metadata)
        end_frame: End frame of original video (for indexing metadata)
        target_height: Target height for intrinsic computation
        target_width: Target width for intrinsic computation
        source_frame_indices: Frame index (0-based int) in original segment for each
                             output frame (from resample_video_frames()). None means
                             one-to-one mapping (default).

    Returns:
        Tuple of pose_dict and action_dict
    """
    actions = metadata.get("actions", {})
    original_total_frames = end_frame - start_frame + 1

    # Intrinsic matrix (FOV = 60 deg, unnormalized; training code normalizes on load)
    focal_length = target_width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
    intrinsic = [
        [focal_length, 0.0, target_width / 2.0],
        [0.0, focal_length, target_height / 2.0],
        [0.0, 0.0, 1.0],
    ]

    # (output frame index, frame index int within original segment)
    if source_frame_indices is None:
        iter_pairs = [(i, i) for i in range(original_total_frames)]
    else:
        iter_pairs = list(enumerate(source_frame_indices))

    pose_dict: Dict = {}
    action_dict: Dict = {}

    for out_idx, frame_offset in iter_pairs:
        # Integer index, look up metadata directly (no interpolation)
        frame_offset = int(frame_offset)
        frame_offset = min(max(0, frame_offset), original_total_frames - 1)
        frame_key = str(start_frame + frame_offset)
        action_data = actions.get(frame_key)

        if action_data is not None:
            w2c = _pose_from_action_data(action_data)
            move_action, view_action = _action_from_action_data(action_data)
        else:
            w2c = np.eye(4, dtype=np.float32)
            move_action = ""
            view_action = ""

        pose_dict[str(out_idx)] = {
            "w2c": w2c.tolist(),
            "intrinsic": intrinsic,
        }
        action_dict[str(out_idx)] = {
            "move_action": move_action,
            "view_action": view_action,
        }

    return pose_dict, action_dict
