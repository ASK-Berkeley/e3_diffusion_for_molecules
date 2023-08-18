#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -t 60-00:00

pwd
hostname
date

source ~/.bashrc
mamba activate edm

export PYTHONUNBUFFERED=1

torchrun --nnodes=1 --nproc_per_node=4 main_geom_drugs.py --n_epochs 100 --n_report_steps 500 --exp_name edm_geom_drugs --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 16 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000000 --resume outputs/edm_geom_drugs

date
