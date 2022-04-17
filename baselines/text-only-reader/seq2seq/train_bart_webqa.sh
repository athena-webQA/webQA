#!/bin/bash
#SBATCH --job-name=train
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




export BS=4
export m=facebook/bart-large
export tok=facebook/bart-large
export MAX_TGT_LEN=128

python finetune_trainer.py \
    --model_name_or_path $m --tokenizer_name $tok \
    --data_dir /home/adityasv/webqa/seq2seq/webqa_dataset \
    --output_dir bart-webqa --overwrite_output_dir \
    --learning_rate=3e-5 \
    --warmup_steps 500 --sortish_sampler \
    --fp16 \
    --n_val 500 \
    --gradient_accumulation_steps=8 \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_embeds \
    --num_train_epochs=2 \
    --save_steps 3000 --eval_steps 3000 \
    --logging_first_step \
    --max_target_length 56 --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN\
    --do_train --do_eval \
    --evaluation_strategy steps \
    --predict_with_generate --sortish_sampler \
    "$@"
