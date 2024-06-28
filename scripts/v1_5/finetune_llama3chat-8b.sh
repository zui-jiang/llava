#!/bin/bash
EXP_ID=$(date +%F-%H-%M-$RANDOM)
bash keys/wandb.sh
export WANDB_RUN_GROUP=IF
export WANDB_ENTITY=cipsup
export WANDB_WATCH=gradients
export WANDB_PROJECT=LLaVA-v1.5
export WANDB_NAME=if-llama3chat-${EXP_ID}
export PYTHONPATH=/ciphome/liuyanjiang2021/LLaVA

deepspeed  --master_port 22922 --include=localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data7/hf_models/NousResearch/Meta-Llama-3-8B-Instruct \
    --version llava_llama_3 \
    --data_path /data5/liuyanjiang2021/hf_datasets/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder /data5/liuyanjiang2021/hf_datasets \
    --vision_tower /data7/hf_models/openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data5/liuyanjiang2021/checkpoints/llama3-chat-8b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data5/liuyanjiang2021/checkpoints/llava-llama3chat-8b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
