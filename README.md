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

`sudo apt install build-essential python3-dev libopenblas-dev`
`pip3 install -r requirements.txt`

Installing MinkowskiEngine:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

To setup the code run the following command on the code main directory:

`pip3 install -U -e .`

## How to Run

A sample dataset can be found [here](https://www.ipb.uni-bonn.de/html/projects/shape_completion/igg_fruit.zip), unzip it and update the paths in the config file `tcore/config/model.yaml`

**Train**  
`python tcore/scripts/train_model.py`

**Test**    
`python tcore/scripts/evaluate_model.py --w <path-to-checkpoint>`  

## How to Cite

If you use this repo, please cite as:

```bibtex  
@inproceedings{magistri2024icra,
author = {F. Magistri and R. Marcuzzi and E.A. Marks and M. Sodano and J. Behley and C. Stachniss},
title = {{Efficient and Accurate Transformer-Based 3D Shape Completion and Reconstruction of Fruits for Agricultural Robots}},
booktitle = {Proc.~of the IEEE Intl.~Conf.~on Robotics \& Automation (ICRA)}, 
year = 2024,
}
