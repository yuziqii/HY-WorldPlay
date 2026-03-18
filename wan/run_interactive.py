"""
WAN 交互式推理脚本 —— 在推理过程中动态改变相机动作

每生成一个 chunk（= 4 latent 帧 = 16 视频帧），脚本暂停并等待用户输入下一步动作，
实现真正的交互式世界模型探索。

运行示例（单 GPU）：
  torchrun --nproc_per_node=1 wan/run_interactive.py \\
      --model_id Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
      --ar_model_path /path/to/wan_transformer \\
      --ckpt_path /path/to/model.pt \\
      --prompt "First-person view walking through a forest" \\
      --image_path /path/to/start.jpg \\
      --out outputs/interactive.mp4

推理中的交互控制（每 chunk 输入一次）：
  w       - 前进
  s       - 后退
  a       - 左平移
  d       - 右平移
  left    - 左转（yaw）
  right   - 右转（yaw）
  up      - 上仰（pitch）
  down    - 下俯（pitch）
  Enter   - 保持上次动作（重复）
  q       - 退出并保存视频

脚本模式（--actions 参数）：
  --actions "w,w,right,w,w,left,w"  逗号分隔的动作序列，每项对应一个 chunk
  脚本模式下不等待输入，适合自动化测试。
"""

import os
import sys
import time
import argparse
import select

import numpy as np
import torch
import imageio
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../wan")))

from wan.inference.helper import MyVAE, CHUNK_SIZE
from wan.inference.pipeline_wan_w_mem_relative_rope import WanPipeline
from wan.models.dits.arwan_w_action_w_mem_relative_rope import WanTransformer3DModel
from wan.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from wan.models.par_vae.tools import DistController
from wan.models.par_vae.context_parallel.wrapper_vae import DistWrapper
from hyvideo.generate_custom_trajectory import rot_x, rot_y


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FORWARD_SPEED = 0.08           # 每 latent 帧前进量（世界单位）
YAW_SPEED = np.deg2rad(3.0)    # 每 latent 帧左/右转（弧度）
PITCH_SPEED = np.deg2rad(3.0)  # 每 latent 帧上/下仰（弧度）

# 归一化内参（对应 1280×704 输出分辨率）
# 原始：fx=fy=969.6969…, cx=960, cy=540
# 归一化：fx /= (cx*2), fy /= (cy*2), cx→0.5, cy→0.5
_FX_NORM = 969.6969696969696 / (960.0 * 2)   # ≈ 0.5052
_FY_NORM = 969.6969696969696 / (540.0 * 2)   # ≈ 0.8977
NORM_INTRINSIC = np.array(
    [[_FX_NORM, 0.0, 0.5],
     [0.0, _FY_NORM, 0.5],
     [0.0, 0.0,      1.0]],
    dtype=np.float64,
)

VALID_ACTIONS = {"w", "s", "a", "d", "left", "right", "up", "down"}

# action label encoding（复用 hyvideo/generate.py 的 mapping）
_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def _one_hot_to_label(one_hot_np):
    """[N, 4] int32 → [N] int label (0-8)"""
    return np.array(
        [_MAPPING.get(tuple(row.tolist()), 0) for row in one_hot_np],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# 相机姿态跟踪器
# ---------------------------------------------------------------------------

class CameraPoseTracker:
    """
    维护交互式推理中相机的全局姿态状态。

    在每次 chunk 调用前，用户提供当前 chunk 的动作指令（一个字符串，
    对所有 CHUNK_SIZE 帧重复应用），生成对应的 viewmats / Ks / action 张量。
    """

    def __init__(self, initial_T: np.ndarray = None):
        """
        参数:
            initial_T: 初始 c2w 矩阵 (4x4)，默认为单位矩阵（世界原点）。
        """
        self.T = initial_T.copy() if initial_T is not None else np.eye(4, dtype=np.float64)
        # _prev_T 保存上一 chunk 最后一帧的 c2w，用于计算第一帧的相对动作标签
        self._prev_T = self.T.copy()

    def _apply_single(self, cmd: str):
        """将单帧动作就地应用到当前 T（c2w 矩阵）。"""
        if cmd == "w":
            self.T[:3, 3] += self.T[:3, :3] @ np.array([0, 0, FORWARD_SPEED])
        elif cmd == "s":
            self.T[:3, 3] += self.T[:3, :3] @ np.array([0, 0, -FORWARD_SPEED])
        elif cmd == "d":
            self.T[:3, 3] += self.T[:3, :3] @ np.array([FORWARD_SPEED, 0, 0])
        elif cmd == "a":
            self.T[:3, 3] += self.T[:3, :3] @ np.array([-FORWARD_SPEED, 0, 0])
        elif cmd == "right":
            self.T[:3, :3] = self.T[:3, :3] @ rot_y(YAW_SPEED)
        elif cmd == "left":
            self.T[:3, :3] = self.T[:3, :3] @ rot_y(-YAW_SPEED)
        elif cmd == "up":
            self.T[:3, :3] = self.T[:3, :3] @ rot_x(PITCH_SPEED)
        elif cmd == "down":
            self.T[:3, :3] = self.T[:3, :3] @ rot_x(-PITCH_SPEED)
        # 其他（未知命令）→ 不动

    def get_chunk_tensors(self, cmd: str, chunk_size: int = CHUNK_SIZE):
        """
        生成当前 chunk 的模型输入张量，同时更新内部姿态状态。

        姿态赋值规则（与 pose_string_to_json 保持一致）：
          - c2ws[0] = 当前 self.T（本 chunk 开始前的位置，无新位移）
          - c2ws[1..chunk_size-1] = 每步应用 cmd 后的新姿态
          - 最后再多应用一次 cmd，将 self.T 推进到下一 chunk 的起始位置

        参数:
            cmd:        动作指令字符串（如 "w", "left"），对整个 chunk 重复应用。
            chunk_size: chunk 的 latent 帧数（默认 CHUNK_SIZE=4）。

        返回:
            viewmats  : torch.Tensor [chunk_size, 4, 4] —— w2c 矩阵
            Ks        : torch.Tensor [chunk_size, 3, 3] —— 归一化内参
            action    : torch.Tensor [chunk_size]       —— 离散动作标签 (0~80)
        """
        # c2ws[0] = 当前位置（不移动），与原始系统的 pose[i*chunk_size] 对应
        c2ws = [self.T.copy()]
        for _ in range(chunk_size - 1):
            self._apply_single(cmd)
            c2ws.append(self.T.copy())

        # 再推进一步，使 self.T 指向下一 chunk 的起始位置
        self._apply_single(cmd)

        c2ws_arr = np.stack(c2ws, axis=0)  # [chunk_size, 4, 4]

        # w2c = inv(c2w)
        w2c_arr = np.linalg.inv(c2ws_arr)  # [chunk_size, 4, 4]

        # 计算相对 c2w，用于推导动作标签
        # relative_c2w[k] = inv(c2ws[k-1]) @ c2ws[k]（k=0 时用 _prev_T）
        # 对 chunk 0 frame 0：_prev_T = identity, c2ws[0] = identity → relative = identity → 静止
        all_c2ws = np.concatenate([self._prev_T[None], c2ws_arr], axis=0)  # [chunk_size+1, 4, 4]
        C_inv = np.linalg.inv(all_c2ws[:-1])    # [chunk_size, 4, 4]
        relative_c2w = C_inv @ all_c2ws[1:]     # [chunk_size, 4, 4]

        action_labels = self._compute_action_labels(relative_c2w)  # [chunk_size]

        Ks = np.tile(NORM_INTRINSIC, (chunk_size, 1, 1))  # [chunk_size, 3, 3]

        # 更新 _prev_T 为本 chunk 最后一帧（用于下一 chunk 第 0 帧的动作标签计算）
        self._prev_T = c2ws_arr[-1].copy()

        return (
            torch.as_tensor(w2c_arr, dtype=torch.float32),
            torch.as_tensor(Ks, dtype=torch.float32),
            torch.as_tensor(action_labels, dtype=torch.float32),
        )

    @staticmethod
    def _compute_action_labels(relative_c2ws: np.ndarray) -> np.ndarray:
        """
        将相对 c2w 矩阵序列编码为离散动作标签 (0~80)。
        标签 = translation_label (0-8) × 9 + rotation_label (0-8)
        """
        n = relative_c2ws.shape[0]
        trans_hot = np.zeros((n, 4), dtype=np.int32)
        rotate_hot = np.zeros((n, 4), dtype=np.int32)

        move_thr = 1e-4
        for i in range(n):
            move_dirs = relative_c2ws[i, :3, 3]
            move_norm = np.linalg.norm(move_dirs)
            if move_norm > move_thr:
                norm_d = move_dirs / move_norm
                ang_rad = np.arccos(norm_d.clip(-1.0, 1.0))
                t_deg = ang_rad * (180.0 / np.pi)
            else:
                t_deg = np.zeros(3)

            R_rel = relative_c2ws[i, :3, :3]
            r = R_scipy.from_matrix(R_rel)
            rot_deg = r.as_euler("xyz", degrees=True)

            if move_norm > move_thr:
                if t_deg[2] < 60:
                    trans_hot[i, 0] = 1   # 前进
                elif t_deg[2] > 120:
                    trans_hot[i, 1] = 1   # 后退
                if t_deg[0] < 60:
                    trans_hot[i, 2] = 1   # 右
                elif t_deg[0] > 120:
                    trans_hot[i, 3] = 1   # 左

            if rot_deg[1] > 5e-2:
                rotate_hot[i, 0] = 1     # 右转
            elif rot_deg[1] < -5e-2:
                rotate_hot[i, 1] = 1     # 左转
            if rot_deg[0] > 5e-2:
                rotate_hot[i, 2] = 1     # 上仰
            elif rot_deg[0] < -5e-2:
                rotate_hot[i, 3] = 1     # 下俯

        trans_label = _one_hot_to_label(trans_hot)
        rot_label = _one_hot_to_label(rotate_hot)
        return trans_label * 9 + rot_label


# ---------------------------------------------------------------------------
# 交互动作输入
# ---------------------------------------------------------------------------

def ask_action(last_cmd: str, rank: int, timeout: float = None) -> str:
    """
    在 rank 0 上读取用户输入的动作指令，其他 rank 从 rank 0 广播接收。

    参数:
        last_cmd:  上次的动作（Enter 时重复使用）
        rank:      当前进程的全局 rank
        timeout:   等待超时秒数（None = 无限等待）

    返回:
        动作字符串，或 None 表示退出。
    """
    cmd = None

    if rank == 0:
        valid_str = "/".join(sorted(VALID_ACTIONS))
        print(f"\n{'─'*50}")
        print(f"下一步动作（{valid_str}）")
        print(f"直接 Enter = 重复上次动作 [{last_cmd}]，输入 q 退出：")
        sys.stdout.flush()

        if timeout is not None:
            # 非阻塞读取（使用 select）
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                line = sys.stdin.readline().strip().lower()
            else:
                print(f"  （超时，自动重复 [{last_cmd}]）")
                line = ""
        else:
            line = input(">>> ").strip().lower()

        if line == "q":
            cmd = "__quit__"
        elif line == "" or line not in VALID_ACTIONS:
            if line != "" and line not in VALID_ACTIONS:
                print(f"  未知指令 '{line}'，保持上次动作 [{last_cmd}]")
            cmd = last_cmd
        else:
            cmd = line

    # 广播给其他 rank
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        # 将 cmd 编码为整数并广播
        action_idx_map = {v: i for i, v in enumerate(sorted(VALID_ACTIONS))}
        action_idx_map["__quit__"] = len(VALID_ACTIONS)

        if rank == 0:
            idx = action_idx_map.get(cmd, action_idx_map[last_cmd])
        else:
            idx = 0

        idx_t = torch.tensor([idx], dtype=torch.int32).cuda()
        torch.distributed.broadcast(idx_t, src=0)
        inv_map = {v: k for k, v in action_idx_map.items()}
        cmd = inv_map[idx_t.item()]

    return cmd


# ---------------------------------------------------------------------------
# 主 Runner
# ---------------------------------------------------------------------------

class InteractiveWanRunner:
    def __init__(self, model_id: str, ckpt_path: str, ar_model_path: str):
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.model_id = model_id
        self.ckpt_path = ckpt_path
        self.ar_model_path = ar_model_path

        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.device = device
        torch.cuda.set_device(self.local_rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        maybe_init_distributed_environment_and_model_parallel(
            1,
            sp_size=self.world_size,
            distributed_init_method="env://",
        )

        self._init_models()

    def _init_models(self):
        if self.rank == 0:
            print("Loading VAE ...")
        self.vae = (
            MyVAE.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
        )

        if self.rank == 0:
            print("Loading WanPipeline ...")
        self.pipe = WanPipeline.from_pretrained(
            self.model_id, vae=self.vae, torch_dtype=torch.bfloat16
        )
        self.pipe.to(self.device)

        dist_controller = DistController(
            self.rank, self.local_rank, self.world_size, None
        )
        dist_vae = DistWrapper(self.vae, dist_controller, None, 4)
        self.pipe.dist_vae = dist_vae

        if self.rank == 0:
            print("Loading AR Transformer ...")
        transformer_ar_action = WanTransformer3DModel.from_pretrained(
            self.ar_model_path,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        transformer_ar_action.add_discrete_action_parameters()

        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        state_dict = state_dict["generator"]
        state_dict = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
        state_dict = {
            k.replace("_fsdp_wrapped_module.", "", 1) if k.startswith("_fsdp_wrapped_module.") else k: v
            for k, v in state_dict.items()
        }
        transformer_ar_action.load_state_dict(state_dict, strict=True)
        self.pipe.transformer = transformer_ar_action.to(dtype=torch.bfloat16)
        self.pipe.to(self.device)

        if self.rank == 0:
            print("Models loaded.")

    def run(
        self,
        prompt: str,
        output_path: str,
        max_chunks: int = 60,
        fps: int = 24,
        height: int = 704,
        width: int = 1280,
        num_inference_steps: int = 4,
        seed: int = 42,
        context_window_length: int = 16,
        image_path: str = None,
        initial_action: str = "w",
        scripted_actions: list = None,
        action_timeout: float = None,
        negative_prompt: str = (
            "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
            "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
            "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
            "杂乱的背景,三条腿,背景人很多,倒着走"
        ),
    ):
        """
        主推理循环。

        参数:
            prompt:           文本提示
            output_path:      输出 MP4 路径
            max_chunks:       最大 chunk 数（= 最大视频时长 / chunk_duration）
            fps:              输出帧率
            height, width:    分辨率
            num_inference_steps: 去噪步数（4=蒸馏快速，50=高质量）
            seed:             随机种子
            context_window_length: 记忆上下文窗口（latent 帧）
            image_path:       起始帧图像（强烈推荐提供）
            initial_action:   第一个 chunk 的默认动作
            scripted_actions: 预定义动作列表（脚本模式）；None = 交互模式
            action_timeout:   每次等待用户输入的超时秒数（None = 无限等待）
        """
        # --- 计算 num_frames（用于管线初始化，需要足够大的缓冲区） ---
        # pipeline 需要 num_frames = (n_latent - 1) * 4 + 1
        # n_latent = max_chunks * CHUNK_SIZE
        n_latent_buffer = max_chunks * CHUNK_SIZE
        num_frames = (n_latent_buffer - 1) * 4 + 1

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"交互式推理初始化")
            print(f"  最大 chunks : {max_chunks}  ({max_chunks * CHUNK_SIZE * 4 / fps:.1f}s @ {fps}fps)")
            print(f"  分辨率      : {width}×{height}")
            print(f"  去噪步数    : {num_inference_steps}")
            print(f"  输出路径    : {output_path}")
            if scripted_actions:
                print(f"  脚本模式    : {scripted_actions}")
            else:
                print(f"  交互模式    : 每 chunk 等待用户输入")
            print(f"{'='*60}\n")

        torch.manual_seed(seed)

        # --- 初始化姿态跟踪器 ---
        pose_tracker = CameraPoseTracker()

        # --- 公共 pipe 参数 ---
        run_args = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            few_step=True,
            first_chunk_size=CHUNK_SIZE,
            return_dict=False,
            image_path=image_path,
            use_memory=True,
            context_window_length=context_window_length,
            output_type="latent",
        )

        # --- 初始化视频写入器 ---
        writer = None
        if self.rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            writer = imageio.get_writer(output_path, fps=fps, quality=8, macro_block_size=1)

        # --- 主推理循环 ---
        total_frames_written = 0
        total_start = time.time()
        last_cmd = initial_action
        scripted_idx = 0

        for chunk_i in range(max_chunks):
            # 1. 获取本 chunk 的动作指令
            if scripted_actions is not None:
                if scripted_idx >= len(scripted_actions):
                    if self.rank == 0:
                        print(f"\n脚本动作已全部执行（{len(scripted_actions)} 个），退出。")
                    break
                cmd = scripted_actions[scripted_idx]
                scripted_idx += 1
                if self.rank == 0:
                    print(f"[脚本模式] Chunk {chunk_i}: 动作 = {cmd}")
            elif chunk_i == 0:
                cmd = initial_action
                if self.rank == 0:
                    print(f"[初始动作] Chunk 0: 动作 = {cmd}")
            else:
                cmd = ask_action(last_cmd, self.rank, timeout=action_timeout)
                if cmd == "__quit__":
                    if self.rank == 0:
                        print("\n用户退出，保存视频...")
                    break

            last_cmd = cmd

            # 2. 生成本 chunk 的 pose 张量
            chunk_viewmats, chunk_Ks, chunk_action = pose_tracker.get_chunk_tensors(cmd, CHUNK_SIZE)

            # 3. 去噪
            chunk_start = time.time()
            self.pipe(
                **run_args,
                chunk_i=chunk_i,
                viewmats=chunk_viewmats.unsqueeze(0).to(self.device),
                Ks=chunk_Ks.unsqueeze(0).to(self.device),
                action=chunk_action.unsqueeze(0).to(self.device),
            )
            denoise_time = time.time() - chunk_start

            # 4. 逐帧解码并写入
            decode_start = time.time()
            for _ in range(CHUNK_SIZE):
                video = self.pipe.decode_next_latent(output_type="np")
                if self.rank == 0 and video is not None:
                    frames = video[0]   # [T, H, W, C]
                    for frame in frames:
                        writer.append_data(frame)
                        total_frames_written += 1
            decode_time = time.time() - decode_start

            elapsed = time.time() - total_start
            if self.rank == 0:
                gen_sec = total_frames_written / fps
                speed = gen_sec / elapsed if elapsed > 0 else 0
                print(
                    f"[{chunk_i+1:3d}/{max_chunks}] 动作={cmd:<6} | "
                    f"去噪 {denoise_time:5.2f}s | 解码 {decode_time:5.2f}s | "
                    f"已生成 {gen_sec:6.2f}s 视频 | 速度 {speed:.3f}x 实时"
                )

        # --- 收尾 ---
        if self.rank == 0 and writer is not None:
            writer.close()
            total_time = time.time() - total_start
            print(f"\n{'='*60}")
            print(f"完成！")
            print(f"总帧数   : {total_frames_written} frames")
            print(f"视频时长 : {total_frames_written / fps:.2f}s")
            print(f"总耗时   : {total_time:.1f}s")
            print(f"输出文件 : {output_path}")
            print(f"{'='*60}")

        return output_path


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WAN 交互式推理 —— 运行时动态改变相机动作")

    # 模型路径
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--ar_model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    # 生成参数
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_window_length", type=int, default=16)

    # 交互控制
    parser.add_argument(
        "--max_chunks", type=int, default=60,
        help="最大 chunk 数（60 chunks × 16帧 × (1/24)s ≈ 40s 视频）",
    )
    parser.add_argument(
        "--initial_action", type=str, default="w",
        help="第一个 chunk 的初始动作（w/s/a/d/left/right/up/down）",
    )
    parser.add_argument(
        "--actions", type=str, default=None,
        help=(
            "脚本模式：逗号分隔的动作序列，每项对应一个 chunk。\n"
            "例：'w,w,right,w,w,left' → 6 个 chunk 的预设动作。\n"
            "若指定此参数，脚本将自动执行而不等待用户输入。"
        ),
    )
    parser.add_argument(
        "--action_timeout", type=float, default=None,
        help="交互模式下等待用户输入的超时秒数（超时后重复上次动作）",
    )

    # 输出
    parser.add_argument("--out", type=str, default="outputs/interactive.mp4")

    args = parser.parse_args()

    # 解析脚本动作
    scripted_actions = None
    if args.actions is not None:
        scripted_actions = [a.strip() for a in args.actions.split(",") if a.strip()]
        for a in scripted_actions:
            if a not in VALID_ACTIONS:
                raise ValueError(
                    f"脚本模式中包含无效动作 '{a}'。"
                    f"有效动作为：{sorted(VALID_ACTIONS)}"
                )

    # 处理可选参数
    kwargs = {}
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt

    runner = InteractiveWanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
    )

    runner.run(
        prompt=args.prompt,
        output_path=args.out,
        max_chunks=args.max_chunks,
        fps=args.fps,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        context_window_length=args.context_window_length,
        image_path=args.image_path,
        initial_action=args.initial_action,
        scripted_actions=scripted_actions,
        action_timeout=args.action_timeout,
        **kwargs,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
