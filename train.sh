#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 0-00:15:00
#SBATCH --mem=64G
#SBATCH -p general
#SBATCH -q debug
#SBATCH --gres=gpu:a100:2
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=npattab1@asu.edu

module load mamba/latest
source activate torch

export CUDA_VISIBLE_DEVICES=1,2
mamba list

echo "executing data parallel"

python mnist_dp.py

echo "starting execution of distributed data parallel"

torchrun --nproc_per_node=4 mnist_ddp.py
