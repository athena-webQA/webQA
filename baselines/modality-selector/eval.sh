#!/bin/bash
#SBATCH --job-name=train-coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:4

python eval_model.py \
    --do_eval \
    --do_predict \
    --train_file /home/adityasv/webqa/modality_selector_training_data/train.json \
    --validation_file /home/adityasv/webqa/modality_selector_training_data/test.json \
    --test_file /home/adityasv/webqa/modality_selector_training_data/test.json \
    --model_name_or_path ./output/ \
    --output_dir ./output/ \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \