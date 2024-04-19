# Efficient and Accurate Transformer-Based 3D Shape Completion and Reconstruction of Fruits for Agricultural Robots

This repo will contain the code for the fruit completion and reconstruction method proposed in our ICRA'24 paper that you can find at this [link](https://www.ipb.uni-bonn.de/pdfs/magistri2024icra.pdf)

![](pics/teaser.png)

The main contribution of this paper is a novel approach
for completing 3D shapes combining template matching
with deep learning. First, we use a 3D sparse convolutional
backbone to extract point-wise features. We then aggregate
such features into vertex features and feed them to a transformer decoder that iteratively deforms our template. Such
an architecture allows us to estimate the complete 3D shape
of fruits when only a partial point cloud is available

## How to Install


Installing python packages pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev` <br>
`pip3 install -r requirements.txt`

Installing MinkowskiEngine:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

To setup the code run the following command on the code main directory:

`pip3 install -U -e .`

## How to Run

**Train**  
`python tcore/scripts/train_model.py`

**Test**    
`python tcore/scripts/evaluate_model.py --w <path-to-checkpoint>`  

## Running our Approach on Sample Data

For running the demo of our approach, we assume that you are using Ubunut 22.04 with a CUDA-capable device, but the scripts can be adapted to other platforms.
We assume that you are in the root directory of the repository. We prepare a small sample dataset (~1.5GB) for testing this repo.

1. Download and extract the sample data: `sh script/dowload_data.sh`
2. Download the checkpoint of our trained model: `sh script/download_checkpoint.sh`

These commands will download the dataset and the checkpoint in `./data/` and `./checkpoints` respectively. 

3. Run the inference on the data: `python tcore/scripts/demo.py --w  checkpoints/pretrained_model.ckpt`

(TODO: add actual commands.)
If you have the Nvidia Container Toolkit installed (see [Setup instructions]()), you can also run the demo as follows:

3. Run our Docker image: `sudo docker run -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/data:/t-core/data -v $(pwd)/checkpoints:/t-core/checkpoints prbonn/t-core python3 tcore/scripts/evaluate_model.py --w  checkpoints/pretrained_model.ckpt`

## Building the Docker image

You can build the Docker image locally via `docker build . -t prbonn/t-core:latest`.

## How to Cite

If you use this repo, please cite as:

```bibtex  
@inproceedings{magistri2024icra,
author = {F. Magistri and R. Marcuzzi and E.A. Marks and M. Sodano and J. Behley and C. Stachniss},
title = {{Efficient and Accurate Transformer-Based 3D Shape Completion and Reconstruction of Fruits for Agricultural Robots}},
booktitle = {Proc.~of the IEEE Intl.~Conf.~on Robotics \& Automation (ICRA)}, 
year = 2024,
}
