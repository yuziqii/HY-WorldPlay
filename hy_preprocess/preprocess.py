"""
数据预处理脚本 - MP4 + NPZ 格式

支持两种模式：
1. 预处理模式（默认）：处理视频和相机数据
2. 检查模式（--check_only）：检查已处理数据的完整性和质量

使用方法：
    # 预处理模式
    python preprocess.py --data_root data.csv --output_dir output --model_path model

    # 检查模式
    python preprocess.py --check_only --input_json output/dataset_index.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from tqdm import tqdm

# Import shared utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils import (
    convert_npz_to_pose_and_actions,
    encode_byt5_prompt,
    encode_first_frame,
    encode_first_frame_to_latent,
    encode_prompt,
    encode_video_to_latent,
    load_byt5_encoder,
    load_text_encoder,
    load_vae_model,
    load_vision_encoder,
    load_video_segment,
    resample_video_frames,
)


# ============================================================================
# 检查模式相关函数
# ============================================================================


def check_and_clean_dataset(
    input_json: str,
    output_json: str,
    min_frames: int = 7,
) -> None:
    """
    检查并清洗已处理的数据集。

    Args:
        input_json: 输入的 dataset_index.json 路径
        output_json: 清洗后输出的 json 路径
        min_frames: 允许的最小 latent 时间帧数
    """
    logger.info(f"加载数据集索引: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    clean_dataset = []
    bad_count = 0
    corrupt_count = 0

    logger.info(f"总计找到 {len(dataset)} 条数据，开始检查 latent 完整性和帧数...")

    for item in tqdm(dataset, desc="Checking latents"):
        latent_path = item.get("latent_path")
        segment_id = item.get("segment_id")

        # 检查文件是否存在
        if not os.path.exists(latent_path):
            logger.warning(f"[丢失] 文件不存在: {latent_path}")
            corrupt_count += 1
            continue

        try:
            # 加载 latent（使用 CPU 避免占用显存）
            pt_data = torch.load(latent_path, map_location="cpu", weights_only=False)

            # 获取 latent tensor
            latent = pt_data["latent"]

            # latent 的 shape 通常是 [1, C, T, H, W] 或 [C, T, H, W]
            # 时间维度 T 在倒数第 3 个位置
            T = latent.shape[-3]

            if T < min_frames:
                logger.debug(f"[太短] 剔除 {segment_id}, 帧数 T={T} (要求 >= {min_frames})")
                bad_count += 1
                continue

            # 检查通过，添加到清洗列表
            clean_dataset.append(item)

        except Exception as e:
            logger.warning(f"[损坏] 无法读取或格式错误 {segment_id}: {e}")
            corrupt_count += 1
            continue

    # 打印统计信息
    logger.info("=" * 50)
    logger.info("清洗完成！")
    logger.info(f"原始数据总量: {len(dataset)}")
    logger.info(f"由于长度太短被剔除的数量: {bad_count}")
    logger.info(f"由于文件损坏或丢失被剔除的数量: {corrupt_count}")
    logger.info(f"最终保留的纯净数据量: {len(clean_dataset)}")
    logger.info("=" * 50)

    # 保存清洗后的数据集
    if len(clean_dataset) > 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(clean_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"纯净版 JSON 已保存至: {output_json}")
        logger.info("请使用这个新的 json 文件进行训练！")
    else:
        logger.warning("没有有效数据，未生成输出文件")


# ============================================================================
# 预处理模式相关函数
# ============================================================================


def load_annotation_csv(csv_path: str) -> List[Dict]:
    """Load annotation CSV with videoFile, cameraFile, caption columns."""
    annotations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(
                {
                    "video_name": row["videoFile"].strip(),
                    "camera_name": row["cameraFile"].strip(),
                    "prompt": row["caption"].strip(),
                }
            )
    return annotations


def preprocess_single_segment(
    video_path: str,
    npz_path: str,
    prompt: str,
    vae: torch.nn.Module,
    text_encoders: Dict,
    vision_encoders: Dict,
    byt5_encoders: Dict,
    output_dir: str,
    segment_id: str,
    target_height: int = 480,
    target_width: int = 832,
    device: str = "cuda",
    target_num_frames: Optional[int] = None,
) -> Dict[str, str]:
    """Preprocess a single video segment with NPZ camera data."""

    segment_output_dir = os.path.join(output_dir, segment_id)
    os.makedirs(segment_output_dir, exist_ok=True)

    # 1. First read the NPZ to know exactly how many camera poses we have
    import numpy as np

    camera_data = np.load(npz_path)
    num_poses = camera_data["extrinsic"].shape[0]

    # 2. Load video segment, capped by the available number of camera poses
    logger.info(f"Loading video: {video_path}")
    video_frames = load_video_segment(video_path, num_frames=num_poses)

    # Just in case video is shorter than NPZ poses
    actual_num_frames = video_frames.shape[0]

    video_frames, source_frame_indices = resample_video_frames(
        video_frames, target_num_frames=target_num_frames
    )
    if target_num_frames is not None:
        logger.info(f"Resampled: {actual_num_frames} -> {video_frames.shape[0]} frames")

    # 3. Encode video to latent
    logger.info("Encoding video to latent...")
    latent = encode_video_to_latent(
        vae, video_frames, target_height=target_height, target_width=target_width, device=device
    )

    # 4. Encode prompts
    logger.info(f"Encoding prompt: {prompt[:50]}...")
    prompt_embeds_dict = encode_prompt(prompt, text_encoders, device=device)
    byt5_embeds_dict = encode_byt5_prompt(prompt, byt5_encoders, device=device)

    # 5. Encode first frame
    logger.info("Encoding first frame for i2v...")
    image_cond = encode_first_frame_to_latent(
        vae, video_frames[0], target_height, target_width, device=device
    )
    vision_states_dict = encode_first_frame(
        video_frames[0], vision_encoders, target_height, target_width, device=device
    )

    # 6. Process pose from NPZ directly
    logger.info("Extracting camera poses from NPZ...")
    pose_dict, action_dict = convert_npz_to_pose_and_actions(
        npz_path, source_frame_indices=source_frame_indices
    )

    # 7. Save to disk
    pose_save_path = os.path.join(segment_output_dir, f"{segment_id}_pose.json")
    with open(pose_save_path, "w") as f:
        json.dump(pose_dict, f, indent=2)

    action_save_path = os.path.join(segment_output_dir, f"{segment_id}_action.json")
    with open(action_save_path, "w") as f:
        json.dump(action_dict, f, indent=2)

    latent_save_path = os.path.join(segment_output_dir, f"{segment_id}_latent.pt")
    save_dict = {
        "latent": latent,
        "prompt_embeds": prompt_embeds_dict["prompt_embeds"],
        "prompt_mask": prompt_embeds_dict["prompt_mask"],
        "byt5_text_states": byt5_embeds_dict["byt5_text_states"],
        "byt5_text_mask": byt5_embeds_dict["byt5_text_mask"],
        "image_cond": image_cond,
        "vision_states": vision_states_dict["vision_states"],
    }
    torch.save(save_dict, latent_save_path)

    return {
        "latent_path": latent_save_path,
        "pose_path": pose_save_path,
        "action_path": action_save_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="数据预处理脚本 - 支持 MP4+NPZ 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预处理模式
  python preprocess.py --data_root data.csv --output_dir output --model_path model

  # 检查模式
  python preprocess.py --check_only --input_json output/dataset_index.json
        """,
    )

    # ========================================================================
    # 模式选择
    # ========================================================================
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="仅检查已处理数据（不进行预处理）",
    )

    # ========================================================================
    # 预处理模式参数
    # ========================================================================
    parser.add_argument("--data_root", type=str, help="Path to the main CSV file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--output_json",
        type=str,
        default="dataset_index.json",
        help="Output index filename",
    )
    parser.add_argument("--model_path", type=str, help="Path to HunyuanVideo model")
    parser.add_argument(
        "--target_height",
        type=int,
        default=480,
        help="Target height (default: 480)",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=832,
        help="Target width (default: 832)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--target_num_frames",
        type=int,
        default=None,
        help="Resample each clip to this frame count",
    )

    # ========================================================================
    # 检查模式参数
    # ========================================================================
    parser.add_argument(
        "--input_json",
        type=str,
        help="Input dataset_index.json for check mode",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=7,
        help="Minimum latent frames for check mode (default: 7)",
    )

    args = parser.parse_args()

    # ========================================================================
    # 检查模式
    # ========================================================================
    if args.check_only:
        if not args.input_json:
            parser.error("--check_only 模式需要 --input_json 参数")

        output_json = args.input_json.replace(".json", "_clean.json")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_json = os.path.join(args.output_dir, os.path.basename(output_json))

        check_and_clean_dataset(
            input_json=args.input_json,
            output_json=output_json,
            min_frames=args.min_frames,
        )
        return

    # ========================================================================
    # 预处理模式
    # ========================================================================
    if not args.data_root:
        parser.error("预处理模式需要 --data_root 参数")
    if not args.output_dir:
        parser.error("预处理模式需要 --output_dir 参数")
    if not args.model_path:
        parser.error("预处理模式需要 --model_path 参数")

    if not args.data_root.endswith(".csv"):
        logger.error(f"--data_root must be a path pointing directly to a CSV file! Got: {args.data_root}")
        sys.exit(1)

    csv_path = args.data_root
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found at {csv_path}")
        sys.exit(1)

    # Derive the base directory (where the csv is) to locate mp4 and npz files
    base_dir = os.path.dirname(csv_path)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Loading models...")
    logger.info("=" * 50)

    vae = load_vae_model(args.model_path, device=args.device)
    text_encoders = load_text_encoder(args.model_path, device=args.device)
    vision_encoders = load_vision_encoder(args.model_path, device=args.device)
    byt5_encoders = load_byt5_encoder(args.model_path, device=args.device)

    logger.info("=" * 50)
    logger.info(f"Loading annotations from {csv_path}")
    logger.info("=" * 50)
    annotations = load_annotation_csv(csv_path)

    if args.num_samples:
        annotations = annotations[: args.num_samples]

    dataset_index = []
    output_json_path = os.path.join(args.output_dir, args.output_json)

    for ann_idx, annotation in enumerate(tqdm(annotations, desc="Processing segments")):
        try:
            video_name = annotation["video_name"]
            camera_name = annotation["camera_name"]
            prompt = annotation["prompt"]

            video_path = os.path.join(base_dir, video_name)
            npz_path = os.path.join(base_dir, camera_name)

            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}, skipping")
                continue

            if not os.path.exists(npz_path):
                logger.warning(f"Camera NPZ not found: {npz_path}, skipping")
                continue

            segment_id = Path(video_name).stem

            result = preprocess_single_segment(
                video_path=video_path,
                npz_path=npz_path,
                prompt=prompt,
                vae=vae,
                text_encoders=text_encoders,
                vision_encoders=vision_encoders,
                byt5_encoders=byt5_encoders,
                output_dir=args.output_dir,
                segment_id=segment_id,
                target_height=args.target_height,
                target_width=args.target_width,
                device=args.device,
                target_num_frames=args.target_num_frames,
            )

            dataset_index.append(
                {
                    "segment_id": segment_id,
                    "video_name": video_name,
                    "video_path": video_path,
                    "latent_path": result["latent_path"],
                    "pose_path": result["pose_path"],
                    "action_path": result["action_path"],
                    "prompt": prompt,
                }
            )
            with open(output_json_path, "w") as f:
                json.dump(dataset_index, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to process segment {ann_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info("=" * 50)
    logger.info(f"Saving dataset index to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(dataset_index, f, indent=2)
    logger.info("Done!")


if __name__ == "__main__":
    main()