#!/bin/bash

EXP_ID=$(date +%F-%H-%M-$RANDOM)
export PYTHONPATH=/ciphome/liuyanjiang2021/LLaVA
export WANDB_API_KEY=67b71a7c534e920d3981223f849e95605d311b84
export WANDB_RUN_GROUP=PRETRAIN
export WANDB_ENTITY=cipsup
export WANDB_WATCH=gradients
export WANDB_PROJECT=LLaVA-Qwen2-1.5B
export WANDB_NAME=resampler-Pretrain-qwen2-1.5b${EXP_ID}

deepspeed --master_port 20121 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data5/liuyanjiang2021/hf_models/Qwen2-1.5B-Instruct \
    --version qwen_2 \
    --data_path /data5/liuyanjiang2021/hf_datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json /ciphome/liuyanjiang2021/llava-toxinic/clean-data/clean-llava_v1_5_mix510k.json \
    --image_folder /data5/liuyanjiang2021/hf_datasets/images \
    --vision_tower /data7/hf_models/openai/clip-vit-large-patch14-336 \
    --mm_projector_type resampler \
    --n_queries 32 \
    --hidden_size_perhead 128 \
    --only_tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data5/liuyanjiang2021/checkpoints/resampler-llava-qwen2instruct-1.5b-32-projectoronly \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
