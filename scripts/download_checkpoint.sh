#!/bin/bash

mkdir checkpoints -p && cd checkpoints
echo Downloading checkpoint ...
wget -O sweetpepper_pretrained.ckpt -c https://www.ipb.uni-bonn.de/html/projects/shape_completion/sweetpepper_pretrained.ckpt
cd ../..

