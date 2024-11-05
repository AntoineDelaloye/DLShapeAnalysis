#!/bin/sh

# Docker mapping
data_folder="/home/jaume/Desktop/Data/Antoine/UKB_Dataset"
output_docker_folder='/usr/data'

code_folder='/home/jaume/Desktop/Code/DLShapeAnalysis'
code_docker_folder='/usr/src'

logs_folder='/home/jaume/Desktop/Logs/DLShapeAnalysis'
log_docker_folder='/usr/logs'

# Interactively
# sudo docker run -v ${data_folder}:${output_docker_folder} -v ${code_folder}:${code_docker_folder} -it localhost:5000/dl_shape:0.0 /bin/bash
# python3 main.py train --config configs/config_UKB_docker.yaml --exp_name test_UKB
# pip uninstall opencv-python -y
# pip install opencv-python==4.8.0.74

sudo docker run \
-v ${data_folder}:${output_docker_folder} \
-v ${code_folder}:${code_docker_folder} \
-v ${logs_folder}:${log_docker_folder} \
localhost:5000/dl_shape:0.0 \
bash -c "cd /usr/src && \
python3 main.py train --config configs/config_UKB_docker.yaml --exp_name test_UKB
"