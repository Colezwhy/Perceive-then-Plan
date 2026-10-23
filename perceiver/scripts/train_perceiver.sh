#!/bin/bash
# Run from perceiver/:
#   TRAIN_JSON=data/omni3d/train.json IMAGE_ROOT=$OMNI3D/datasets bash scripts/train_perceiver.sh
set -e

MODEL=${MODEL:-Qwen/Qwen3-VL-2B-Instruct}        # or Qwen/Qwen3-VL-8B-Instruct
TRAIN_JSON=${TRAIN_JSON:?set TRAIN_JSON to the file written by tools/prepare_omni3d.py}
IMAGE_ROOT=${IMAGE_ROOT:?set IMAGE_ROOT to the directory containing SUNRGBD/ and hypersim/}
OUTPUT_DIR=${OUTPUT_DIR:-output/perceiver-$(basename ${MODEL})}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-$((16 / NPROC_PER_NODE))}     # keeps the global batch at 16
ATTN=${ATTN:-flash_attention_2}                        # sdpa if flash-attn is not installed
REPORT_TO=${REPORT_TO:-none}                           # e.g. wandb

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

export PYTHONPATH=$(pwd):${PYTHONPATH}

torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ptp_perceiver/train/train_perceiver.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL} \
    --train_json ${TRAIN_JSON} \
    --image_root ${IMAGE_ROOT} \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --geometry_encoder_type vggt \
    --geometry_encoder_path facebook/VGGT-1B \
    --geometry_merger_type mlp \
    --feature_fusion_method film \
    --attn_implementation ${ATTN} \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_pixels $((576 * 32 * 32)) \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --seed 42 \
    --report_to ${REPORT_TO}
