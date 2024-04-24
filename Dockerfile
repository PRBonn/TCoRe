FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV PATH /opt/conda/bin:$PATH

COPY . /t-core
WORKDIR /t-core

SHELL ["/bin/bash","-l", "-c"]

RUN apt update && apt install --assume-yes --no-install-recommends build-essential python3-dev libopenblas-dev python3-pip libegl1 libgl1 libgomp1 wget git unzip && rm -rf /var/lib/apt/lists/*
# Adapted from https://hub.docker.com/r/continuumio/miniconda3/dockerfile
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN conda init && . ~/.bashrc # '.' used instead of 'source' to load commands from .bashrc 

RUN conda create --name tcore python=3.9 && conda activate

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "tcore", "/bin/bash", "-c"] ## only needed inside docker.

RUN pip3 install pip==23.0.0
RUN pip3 install -r requirements.txt
RUN pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-options="--force-cuda"
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip3 install -U -e .
