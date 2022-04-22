#!/bin/bash
#SBATCH --job-name=train-coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:4

python run_glue_no_trainer.py \
    --train_file /home/adityasv/webqa/modality_selector_training_data/train.json \
    --validation_file /home/adityasv/webqa/modality_selector_training_data/valid.json \
    --model_name_or_path bert-base-uncased \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --weight_decay 1e-5 \
    --num_train_epochs 10 \
    --output_dir ./output/