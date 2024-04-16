FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY . /t-core
WORKDIR /t-core

RUN apt update
RUN apt install --assume-yes --no-install-recommends build-essential python3-dev libopenblas-dev python3-pip
RUN pip3 install -r requirements.txt
RUN pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
RUN pip3 install -U -e .
