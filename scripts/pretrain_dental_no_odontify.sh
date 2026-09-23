#!/bin/sh
set -e

# MAE 牙科 SSL 预训练 —— no-odontify 病例级 split
#（自有采集者 + AIDC 纳入训练；odontify 外部集整体排除）
#
# 独立脚本：自带全部训练超参与 GPU / 端口配置，直接调 torchrun。
#
# 用法（在 baselines/mae 下执行；从任意目录运行也可，路径自动解析）:
#   ./scripts/pretrain_dental_no_odontify.sh
#   ./scripts/pretrain_dental_no_odontify.sh --resume exp/mae_dental_448_no_odontify/checkpoint-050.pth
#   ./scripts/pretrain_dental_no_odontify.sh --init_checkpoint <adapted-ckpt>
#
# 可覆盖环境变量:
#   GPUS        默认 6 卡 GPU 0-5（分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀）
#               GPUS="0 1" / GPUS="0,1" / GPUS="cuda:0 cuda:1" 三者等价
#   INPUT_SIZE  默认 448（与 ours full_canvas=448 统一）
#   DATA_ROOT   默认 <root>/.datasets/intraoral（挂载点缺失时回退 intraoral1）
#   SPLIT_JSON  默认 .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   OUTPUT_DIR  默认 baselines/mae/exp/mae_dental_448_no_odontify
#   INIT_CKPT   默认 .checkpoints/intraoral/mae_pretrain_vit_base_full.pth
#   MASTER_PORT 默认 29531（torchrun 默认 29500 常被其它任务占用）
#
# 分辨率说明：默认 448，dense 特征网格 28x28（patch 16）。官方
#   mae_pretrain_vit_base_full.pth（224）初始化时 util/pos_embed.py::interpolate_pos_embed
#   会自动把 pos_embed 从 16x16 双三次插值到 32x32（448/16），无需手工处理。
#   需要 224 时：INPUT_SIZE=224 ./scripts/pretrain_dental_no_odontify.sh
#
# 前置（一次性，约 40s；已存在则不会覆盖）:
#   python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \
#       --seed 0 --test_ratio 0.2 \
#       --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json
#   预期: sample_ids_total=2909 / annotation_json_total=11642 / annotation_json_excluded=891
#         train 2327 case (9317 json) / test 582 case (2325 json)
export OMP_NUM_THREADS=1

# 目录定位（脚本位于 <root>/baselines/mae/scripts/）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${MAE_ROOT}/../.." && pwd)"

# 指定用于训练的 GPU：分隔符空格 / 逗号 / 分号都接受，元素可带 cuda: 前缀
#   GPUS="0 1 2"   GPUS="0,1,2"   GPUS="cuda:0 cuda:1"   GPUS="0, 1"
# 默认与 scripts/pretrain_dental.sh 保持一致（0-5）
GPUS="${GPUS:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5n}"

# 解析 GPUS -> torchrun 用的 CUDA_VISIBLE_DEVICES 列表与卡数 NUM_GPUS
# set -f：解析期间关闭通配符展开（GPUS 里误写 * ? [ 时不会被 glob 成文件名）。
CUDA_VISIBLE_DEVICES=""
NUM_GPUS=0
set -f
for GPU in $(printf '%s\n' "${GPUS}" | tr ',;' '  '); do
    GPU_ID=${GPU#cuda:}
    case "${GPU_ID}" in
        ''|*[!0-9]*)
            echo "[pretrain_dental_no_odontify] ERROR: GPUS 中的 '${GPU}' 不是合法 GPU 序号" >&2
            echo "  合法写法: GPUS=\"2 3\" / GPUS=\"2,3\" / GPUS=\"cuda:2 cuda:3\"" >&2
            exit 1
            ;;
    esac
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES},${GPU_ID}
    fi
    NUM_GPUS=$((NUM_GPUS + 1))
done
set +f
if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "[pretrain_dental_no_odontify] ERROR: GPUS 为空，应形如 GPUS=\"2 3\" 或 GPUS=\"2,3\"" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES

# 分布式 rendezvous 端口：避开常被占用的 29500
MASTER_PORT="${MASTER_PORT:-29531}"

# 训练输入分辨率（与 ours full_canvas=448 统一）；ViT 全局注意力 O(n^2)，
# 448 下 patch token 为 28x28=784（224 时 196），batch_size 已相应调小（224 时为 64）
INPUT_SIZE="${INPUT_SIZE:-448}"

# 数据根目录兼容：优先 .datasets/intraoral，找不到（挂载点缺失 / 符号链接失效）时
# 回退到 .datasets/intraoral1。显式设置 DATA_ROOT / SPLIT_JSON 时以环境变量为准，不回退。
resolve_dental_path() {
    # $1 = 相对 .datasets 层名的后缀（"" 或 "/train_test_seed0_test0p2_no_odontify.json"）
    for ROOT_NAME in intraoral intraoral1; do
        if [ -e "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1" ]; then
            printf '%s\n' "${WORKSPACE_ROOT}/.datasets/${ROOT_NAME}$1"
            return 0
        fi
    done
    printf '%s\n' "${WORKSPACE_ROOT}/.datasets/intraoral$1"
}

DATA_ROOT="${DATA_ROOT:-$(resolve_dental_path "")}"
SPLIT_JSON="${SPLIT_JSON:-$(resolve_dental_path "/train_test_seed0_test0p2_no_odontify.json")}"
# 用独立输出目录，避免覆盖 compare/ manifest 仍在引用的常规 split 产物
OUTPUT_DIR="${OUTPUT_DIR:-${MAE_ROOT}/exp/mae_dental_448_no_odontify}"
INIT_CKPT="${INIT_CKPT:-${WORKSPACE_ROOT}/.checkpoints/intraoral/mae_pretrain_vit_base_full.pth}"

if [ ! -e "${SPLIT_JSON}" ]; then
    echo "[pretrain_dental_no_odontify] ERROR: split 不存在: ${SPLIT_JSON}" >&2
    echo "  先生成（一次性，约 40s）:" >&2
    echo "    python datasets/split.py --data_root .datasets/intraoral --exclude_dirs odontify \\" >&2
    echo "        --seed 0 --test_ratio 0.2 \\" >&2
    echo "        --split_json .datasets/intraoral/train_test_seed0_test0p2_no_odontify.json" >&2
    exit 1
fi

# 额外透传参数（如 --resume ...、--category_sampling ...）
EXTRA_ARGS="$@"

cd "${MAE_ROOT}"
echo "[pretrain_dental_no_odontify] GPUS=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
echo "[pretrain_dental_no_odontify] DATA_ROOT=${DATA_ROOT}"
echo "[pretrain_dental_no_odontify] SPLIT_JSON=${SPLIT_JSON}"
echo "[pretrain_dental_no_odontify] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[pretrain_dental_no_odontify] MASTER_PORT=${MASTER_PORT}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" main_pretrain.py \
    --accum_iter 2 \
    --input_size "${INPUT_SIZE}" \
    --batch_size 16 \
    --model mae_vit_base_patch16 \
    --epochs 100 \
    --warmup_epochs 5 \
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
