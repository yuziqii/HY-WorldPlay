
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 路径配置（TRAIN_JSON 和 OUTPUT_PATH 需手动填写）
# ============================================================
TRANSFORMER_PATH=$MODEL_PATH/transformer/480p_i2v

MODEL_PATH=                   # Path to pretrained hunyuanvideo-1.5 model
AR_ACTION_MODEL_PATH=         # Path to our HY-World 1.5 autoregressive checkpoints
AR_DISTILL_ACTION_MODEL_PATH= # Path to our HY-World 1.5 autoregressive distilled checkpoints
TRAIN_JSON=                   # 训练数据 json 文件路径（必填）
OUTPUT_PATH=./outputs/ckpt
# ============================================================
# LoRA 超参数
# rank 越大容量越强但显存越多，域适应建议 16~64
# alpha 通常设置为与 rank 相同，控制 LoRA 缩放强度
# ============================================================
LORA_RANK=32
LORA_ALPHA=32

# ============================================================
# 并行配置
# LoRA 参数量小，sp_size=4 在 4 卡上即可运行
# ============================================================
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ============================================================
# 训练参数
# 域微调不需要全量训练步数，2000~5000 步通常足够收敛
# ============================================================
training_args=(
  --data-path /tmp
  --json_path $TRAIN_JSON
  --causal
  --action
  --i2v_rate 0.2
  --train_time_shift 3.0
  --window_frames 24
  --tracker_project_name basketball_lora_finetune
  --OUTPUT_PATH $OUTPUT_PATH
  --max_train_steps 3000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 2
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
)

# ============================================================
# LoRA 专用参数（核心开关）
# lora_target_modules 默认为全部注意力层：
#   q_proj, k_proj, v_proj, o_proj, to_q, to_k, to_v, to_out, to_qkv
# ============================================================
lora_args=(
  --lora-training True
  --lora-rank $LORA_RANK
  --lora-alpha $LORA_ALPHA
)

# ============================================================
# 并行参数
# ============================================================
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 4
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# ============================================================
# 模型参数
# ============================================================
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir $TRANSFORMER_PATH
  --ar_action_load_from_dir $AR_ACTION_MODEL_PATH
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

dataset_args=(
  --dataloader_num_workers 1
)

validation_args=(
  --validation_steps 500
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

# LoRA 微调用更高学习率（比全量微调高 5~10 倍）
optimizer_args=(
  --learning_rate 1e-4
  --mixed_precision "bf16"
  --checkpointing_steps 500
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 5
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

export MASTER_PORT=29613

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NUM_GPUS \
  --nnodes 1 \
  trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${lora_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${miscellaneous_args[@]}"
