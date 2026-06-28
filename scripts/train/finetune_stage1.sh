#!/bin/bash
# ============================================================================
# Stage 1 Training Script
# Runs the first stage of continual learning on dataset stage1.
# No previous task model needed. Output to checkpoints/.
# ============================================================================
export OMP_NUM_THREADS=16
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="qwen_llava7B_llm_cladmoe_anomaly_stage1"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 1
PROMPT_VERSION="qwen_1_5"
RUN_NAME="qwen_llava7B_llm_cladmoe_anomaly_stage1"
PREV_STAGE_CHECKPOINT="/data1/dongwen_data/Model/llava-onevision-qwen2-7b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

NUM_GPUS=8
NNODES=1
RANK=0
ADDR=127.0.0.1
PORT=29505

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29600 scripts/train/train_mem_anomaly.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --lora_enable True --lora_r 36 --lora_alpha 72 --mm_projector_lr 2e-5 \
    --expert_num 12 --private_expert_num 8 \
    --cur_task 0 \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /home/dongwen/CL-Anomaly/data/datasets_stage1.yaml \
    --image_folder /data1/dongwen_data/CL-Anomaly/datasets \
    --video_folder None \
    --mm_vision_tower_lr=1e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /data1/dongwen_data/CL-Anomaly/checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --task $RUN_NAME
exit 0;
