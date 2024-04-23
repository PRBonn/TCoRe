FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV PATH /opt/conda/bin:$PATH

COPY . /t-core
WORKDIR /t-core

RUN apt update && apt install --assume-yes --no-install-recommends build-essential python3-dev libopenblas-dev python3-pip libegl1 libgl1 libgomp1 wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh && bash /opt/conda/miniconda.sh -b -p /opt/miniconda 

RUN conda create --name tcore python=3.9 && conda activate tcore
RUN pip3 install -r requirements.txt
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
RUN pip3 install -U -e .
