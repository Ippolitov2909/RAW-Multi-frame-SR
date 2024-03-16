# RAW-Multi-frame-SR
## Overview
This repository contains my own implementation of image processing pipeline that is applied to RAW images to obtain RGB images seen by smartphone users. The whole algorithm can be observed as joint multi-frame demosaicing, denoising and super-resolution. 

Most of implemented code is inspired by Google's paper ["Handheld multi-frame super-resolution"](https://sites.google.com/view/handheld-super-res/): 

<p align="center">
  <img src="https://github.com/Ippolitov2909/RAW-Multi-frame-SR/blob/main/examples/orig_piece.png" width="300" align="middle"/>
  <img src="https://media.istockphoto.com/id/1093587922/vector/arrow-line-vector-icon-on-white-background.jpg?s=612x612&w=0&k=20&c=oMajz7P9e9tqTxW4aFjTAejxuydaQ9nHdPo6Ht5B69E=" width="200" align="middle"/>
  <img src="https://github.com/Ippolitov2909/RAW-Multi-frame-SR/blob/main/examples/merged_piece.png" width="300" align="middle" /> 
</p>

## Pipeline
*Input*: burst of RAW images
A burst of RAW images are processed in the following order. 

1) Calculating optical flow between each frame and the base frame

2) Calculating merge kernels --- kernels that are used to merge all frames into one

3) Calculating robustness mask --- mask that is used to mask unreliable pixels so that they are not used in merging

4) Merging all frames into one using all obtained information

*Output*: one RGB denoised image with x2 higher resolution

## Usage
Repository contains 6 jupyter notebooks. Each of them has examples of running my code

* Opening_raw.ipynb --- how to open burst of RAW images
* Optical_flow.ipynb --- how to estimate optical flow using OpticalFlowProcessor with Lucas-Kanade method inside it
* Kernel_resgression.ipynb --- how to estimate merge kernels
* Robustness.ipynb --- how to estimate robustness mask
* Merging.ipynb --- how to merge all RAW frames into one on original resolution
* Superresolution.ipynb --- how to merge all RAW frames into one on x2 resolution
