#!/bin/bash
#SBATCH --mem 64000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --nodelist=boston-2-27

export CUDA_LAUNCH_BLOCKING=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 

python -m torch.distributed.run --nproc_per_node=2 train_retrieval_webqa.py \
--config ./configs/retrieval_webqa.yaml \
--output_dir output/retrieval_webqa_pretrained
