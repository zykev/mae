#!/bin/sh
set -e

# MAE 牙科 SSL 预训练（病例级 split，§9.5.3）
# 用法（在 baselines/mae 下执行；从任意目录运行也可，路径自动解析）:
#   ./scripts/pretrain_dental.sh                    # 从头训练（官方权重初始化）
#   ./scripts/pretrain_dental.sh --resume exp/mae_dental_448/checkpoint-050.pth   # 中断恢复
#   ./scripts/pretrain_dental.sh --init_checkpoint <adapted-ckpt>                # 权重初始化
#
# 分辨率说明：默认 INPUT_SIZE=448，与 ours（full_canvas 448）统一，dense 特征网格
#   28x28（patch 16）。官方 mae_pretrain_vit_base_full.pth（224）初始化时
#   util/pos_embed.py::interpolate_pos_embed 会自动把 pos_embed 从 16x16 双三次插值
#   到 32x32（448/16），无需手工处理。
#   需要 224 时：INPUT_SIZE=224 ./scripts/pretrain_dental.sh
# 分布式 rendezvous 端口：本机常有其它任务（torchrun/accelerate 默认 29500）长期占用，
# 故默认避开 29500；冲突时可用 MASTER_PORT=<port> ./scripts/pretrain_dental.sh 覆盖。
export OMP_NUM_THREADS=1

# 目录定位（脚本位于 <root>/baselines/mae/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${MAE_ROOT}/../.." && pwd)"

# 指定用于训练的 GPU，可以写成 "cuda:0 cuda:1 cuda:2" 或 "0 1 2"
GPUS="${GPUS:-cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7}"

# 训练输入分辨率（与 ours full_canvas=448 统一）；ViT 全局注意力 O(n^2)，
# 448 下 patch token 为 28x28=784（224 时 196），batch_size 已相应调小（224 时为 64）
INPUT_SIZE="${INPUT_SIZE:-448}"

CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
for GPU in $GPUS; do
    GPU_ID=${GPU#cuda:}
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID
    else
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,$GPU_ID
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done

export CUDA_VISIBLE_DEVICES

DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/.datasets/intraoral}"
SPLIT_JSON="${SPLIT_JSON:-${WORKSPACE_ROOT}/.datasets/intraoral/train_test_seed0_test0p2.json}"
# 448 分辨率用独立输出目录，避免覆盖已有 224 训练产物（compare manifest 引用旧路径）
OUTPUT_DIR="${OUTPUT_DIR:-${MAE_ROOT}/exp/mae_dental_448}"
INIT_CKPT="${INIT_CKPT:-${WORKSPACE_ROOT}/.checkpoints/intraoral/mae_pretrain_vit_base_full.pth}"

# 额外透传参数（如 --resume ...、--category_sampling ...）
EXTRA_ARGS="$@"

cd "${MAE_ROOT}"
echo "[pretrain_dental] GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[pretrain_dental] DATA_ROOT=${DATA_ROOT}"
echo "[pretrain_dental] SPLIT_JSON=${SPLIT_JSON}"
echo "[pretrain_dental] OUTPUT_DIR=${OUTPUT_DIR}"

MASTER_PORT="${MASTER_PORT:-29501}"
echo "[pretrain_dental] MASTER_PORT=${MASTER_PORT}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" main_pretrain.py \
    --accum_iter 2 \
    --input_size "${INPUT_SIZE}" \
    --batch_size 16 \
    --model mae_vit_base_patch16 \
    --epochs 200 \
    --warmup_epochs 20 \
    --mask_ratio 0.75 \
    --blr 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --vis_freq 50 \
    --vis_num_images 8 \
    --norm_pix_loss \
    --data_path "${DATA_ROOT}" \
    --split_json "${SPLIT_JSON}" \
    --split train \
    --categories "full,tooth,sextant,group" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    --init_checkpoint "${INIT_CKPT}" \
    ${EXTRA_ARGS}
