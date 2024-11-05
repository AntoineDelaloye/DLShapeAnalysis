#!/bin/sh

sudo docker build -f Dockerfile -t localhost:5000/dl_shape:0.0 .
sudo docker push localhost:5000/dl_shape:0.0

# Singularity images path
# SINGULARITY_IMGS="/home/jaume/Desktop/Data/SingularityImages"
# rm -rfd ${SINGULARITY_IMGS}/dl_shape_0.0.sif
# cd ${SINGULARITY_IMGS}
# SINGULARITY_NOHTTPS=1 singularity pull docker://localhost:5000/dl_shape:0.0