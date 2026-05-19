#!/bin/bash
set -e

# 设置环境变量
export OMP_NUM_THREADS=1

# 指定用于训练的 GPU，可以写成 cuda:0 cuda:1 cuda:2 或 0 1 2
GPUS=(cuda:0 cuda:1 cuda:2)

GPU_IDS=()
for GPU in "${GPUS[@]}"; do
    GPU_IDS+=("${GPU#cuda:}")
done

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
export CUDA_VISIBLE_DEVICES
NUM_GPUS=${#GPU_IDS[@]}

# 执行 Python 程序
torchrun --nproc_per_node="${NUM_GPUS}" main_pretrain.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --resume ".checkpoints/intraoral/mae_pretrain_vit_base_full.pth" \
    --epochs 400 \
    --warmup_epochs 40 \
    --mask_ratio 0.75 \
    --blr 5e-5 \
    --weight_decay 0.05 \
    --data_path ".datasets/intraoral" \
    --output_dir "exp/pretrain_v3" \
    --log_dir "exp/pretrain_v3"
