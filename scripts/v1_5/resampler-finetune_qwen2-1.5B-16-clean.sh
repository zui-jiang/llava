#!/bin/bash

EXP_ID=$(date +%F-%H-%M-$RANDOM)
export PYTHONPATH=/ciphome/liuyanjiang2021/LLaVA
export WANDB_API_KEY=67b71a7c534e920d3981223f849e95605d311b84
export WANDB_RUN_GROUP=IF
export WANDB_ENTITY=cipsup
export WANDB_WATCH=gradients
export WANDB_PROJECT=LLaVA-Qwen2-1.5B
export WANDB_NAME=resampler-if-Qwen2-1.5B${EXP_ID}

deepspeed  --master_port 20522  llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /data5/liuyanjiang2021/hf_models/Qwen2-1.5B-Instruct \
    --version qwen_2 \
    --data_path /ciphome/liuyanjiang2021/llava-toxinic/clean-data/clean-llava_v1_5_mix510k.json \
    --image_folder /data5/liuyanjiang2021/hf_datasets \
    --vision_tower /data7/hf_models/openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data5/liuyanjiang2021/checkpoints/resampler-llava-qwen2instruct-1.5b-pretrain-16-clean/mm_projector.bin \
    --mm_projector_type resampler \
    --n_queries 16 \
    --hidden_size_perhead 128 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data5/liuyanjiang2021/checkpoints/resampler-llava-qwen2instruct-1.5b-16-clean \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
