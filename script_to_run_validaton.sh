#!/bin/bash
#SBATCH --job-name=dl_shape
##SBATCH --output=/cluster/home/an1979/logs/dl_shape_%A_%a.out
##SBATCH --error=/cluster/home/an1979/logs/dl_shape_%A_%a.err
#SBATCH --output=/cluster/home/ja1659/logs/dl_shape_%A_%a.out
#SBATCH --error=/cluster/home/ja1659/logs/dl_shape_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=rad
#SBATCH --account=rad

# Singularity folder
singularity_path='/data/bdip2/jbanusco/SingularityImages'
singularity_img=${singularity_path}/dl_shape_0.0.sif

# Dataset path
# dataset_path="/data/bdip2/an1979/UKB_Dataset"
dataset_path="/data/bdip2/jbanusco/UKB_Cardiac_BIDS"
# Code path
code_path='/cluster/home/ja1659/Code/DLShapeAnalysis'
# code_path='/cluster/home/an1979/DLShapeAnalysis'

# Logs path
logs_folder=${dataset_path}/derivatives/dl_shape_logs
# mkdir -p ${logs_folder}

# Docker mapping
docker_data='/usr/data'
docker_code='/usr/src'
docker_log='/usr/logs'

# Test singularity docker
singularity exec --nv \
--bind ${dataset_path}:${docker_data} \
--bind ${code_path}:${docker_code} \
--bind ${logs_folder}:${docker_log} \
${singularity_img} /bin/bash -c "cd ${docker_code} && python3 main.py eval --config configs/config_UKB_docker.yaml --exp_name eval_UKB"