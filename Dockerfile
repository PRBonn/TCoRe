FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY . /t-core
WORKDIR /t-core

RUN apt update && apt install --assume-yes --no-install-recommends build-essential python3-dev libopenblas-dev python3-pip libegl1 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip3 install -r requirements.txt
RUN pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
RUN pip3 install -U -e .

## todo: 
## 1. Provide a script for downloading the data into a convienient local directory, i.e., `data/fruits_igg_test/` or something
## 2. Try running the pipeline in the Docker container

## Is there actually also some visual output or just a number?
