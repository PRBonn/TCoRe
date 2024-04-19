#!/bin/bash

mkdir checkpoints -p
cd checkpoints

echo Downloading checkpoint ...
wget -O pretrained_model.ckpt -c https://www.ipb.uni-bonn.de/html/projects/shape_completion/sweetpepper_pretrained.ckpt

cd ../..

