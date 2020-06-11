#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p t4
#SBATCH --gres=gpu:1
#SBATCH --qos=normal            # run mpi tasks on qos high to reduce risk of preemption 
#SBATCH --mem=16G               # amount of memory to allocate per node
#SBATCH --gpus-per-task=1       # number of GPUs we want per task
#SBATCH --cpus-per-gpu=4        # number of CPUs we want per GPU
#SBATCH --job-name=one_epoch
#SBATCH --output=%x.out

python /h/yuchen/cluster/model.py -ne 1
