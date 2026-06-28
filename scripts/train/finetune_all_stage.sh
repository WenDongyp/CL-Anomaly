#!/bin/bash
# ============================================================================
# Continual Learning Full Pipeline (8 Stages)
#
# Pipeline per stage:
#   Stage 1:         [train] → [router stats]
#   Stage 2-8:       [train] → [router stats] → [LoRA merge]
#
# Merged stage files:
#   scripts/train/finetune_stage1.sh         - Stage 1 training
#   scripts/train/finetune_stage.sh <STAGE>  - Stage 2-8 training (parametrized)
# ============================================================================
set -e

RUN_PREFIX="qwen_llava7B_llm_cladmoe_anomaly"

# ========================= Stage 1 =========================
echo ""
echo "========== Stage 1 =========="
bash scripts/train/finetune_stage1.sh
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage1

# ========================= Stage 2 =========================
echo ""
echo "========== Stage 2 =========="
bash scripts/train/finetune_stage.sh 2
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage2
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage1 --task2 ${RUN_PREFIX}_stage2

# ========================= Stage 3 =========================
echo ""
echo "========== Stage 3 =========="
bash scripts/train/finetune_stage.sh 3
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage3
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage2 --task2 ${RUN_PREFIX}_stage3

# ========================= Stage 4 =========================
echo ""
echo "========== Stage 4 =========="
bash scripts/train/finetune_stage.sh 4
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage4
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage3 --task2 ${RUN_PREFIX}_stage4

# ========================= Stage 5 =========================
echo ""
echo "========== Stage 5 =========="
bash scripts/train/finetune_stage.sh 5
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage5
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage4 --task2 ${RUN_PREFIX}_stage5

# ========================= Stage 6 =========================
echo ""
echo "========== Stage 6 =========="
bash scripts/train/finetune_stage.sh 6
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage6
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage5 --task2 ${RUN_PREFIX}_stage6

# ========================= Stage 7 =========================
echo ""
echo "========== Stage 7 =========="
bash scripts/train/finetune_stage.sh 7
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage7
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage6 --task2 ${RUN_PREFIX}_stage7

# ========================= Stage 8 =========================
echo ""
echo "========== Stage 8 =========="
bash scripts/train/finetune_stage.sh 8
CUDA_VISIBLE_DEVICES=3 python scripts/train/statistic.py --task ${RUN_PREFIX}_stage8
CUDA_VISIBLE_DEVICES=3 python scripts/train/lora_param_merge.py --task1 ${RUN_PREFIX}_stage7 --task2 ${RUN_PREFIX}_stage8

echo ""
echo "========== ALL 8 STAGES COMPLETED =========="
