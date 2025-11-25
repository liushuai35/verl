export CUDA_HOME=/home/a205/anaconda3/envs/ls
export PYTHONPATH= /home/a205/anaconda3/envs/ls/bin/python
unset PYTHONPATH

/mnt/PublicStorageNew1/liushuai/dataset/segmented_glm


# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-7B-Instruct \
    --train_type lora \
    --dataset 'local/segmented_glm' 
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot


CUDA_VISIBLE_DEVICES=0,1,2 NPROC_PER_NODE=3 \
swift sft \
--torch_dtype 'bfloat16' \
--model 'Qwen/Qwen3-8B-Base' \
--model_type 'qwen3' \
--template 'qwen3' \
--dataset 'local/segmented_glm' \
--split_dataset_ratio '0.1' \
--max_length '4096' \
--task_type 'causal_lm' \
--lora_dtype 'bfloat16' \
--per_device_train_batch_size '1' \
--per_device_eval_batch_size '1' \
--learning_rate '1e-5' \
--num_train_epochs '10' \
--truncation_strategy left \
--gradient_accumulation_steps '8' \
--eval_steps '100' \
--save_steps '200' \
--attn_impl 'flash_attention_2' \
--neftune_noise_alpha '0' \
--report_to 'wandb' \
--deepspeed zero1 \
--add_version False \
--output_dir /mnt/PublicStorageNew1/liushuai/ms-swift/output/Qwen3-8B-Base/v0-20251125-210325 \
--logging_dir /mnt/PublicStorageNew1/liushuai/ms-swift/output/Qwen3-8B-Base/v0-20251125-210325/runs \
--ignore_args_error True 