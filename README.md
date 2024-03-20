# Prediction in latent space

Code of "Control guided by Prediction in Latent Space" part for the paper accepted in 2023IROS "Robust Real-Time Motion Retargeting via Neural Latent Prediction" [here](https://ieeexplore.ieee.org/abstract/document/10342022)

## Prerequisite

- [**PyTorch**](https://pytorch.org/get-started/locally/) Tensors and Dynamic neural networks in Python with strong GPU acceleration
- [**pytorch_geometric**](https://github.com/rusty1s/pytorch_geometric) Geometric Deep Learning Extension Library for PyTorch
- [**Kornia**](https://github.com/kornia/kornia) a differentiable computer vision library for PyTorch.
- [**HDF5 for Python**](https://docs.h5py.org/en/stable/) The h5py package is a Pythonic interface to the HDF5 binary data format.


## Dataset

The Chinese sign language dataset can be downloaded [here](https://www.jianguoyun.com/p/DYm5RzMQ74eHChj_lJ0E).

## Model

The pretrained prediction model can be downloaded [here](https://www.jianguoyun.com/p/Dc6_7pkQosW7DBiayrsFIAA)

## Get Started

**Training**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg './configs/train/yumi.yaml'
```

**Inference**

inference one key:
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg './configs/inference/yumi.yaml' 
```

inference all keys:
```bash
CUDA_VISIBLE_DEVICES=0 python inference_all.py --cfg './configs/inference/yumi.yaml' 
```


