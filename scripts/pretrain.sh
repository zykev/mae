#!/bin/bash

# 设置环境变量
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# 执行 Python 程序
python main_pretrain.py \
    --accum_iter 2 \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --resume ".checkpoints/intraoral/mae_pretrain_vit_base_full.pth" \
    --epochs 400 \
    --warmup_epochs 20 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --data_path ".datasets/intraoral/intraoral" \
    --output_dir "exp/pretrain_v1"