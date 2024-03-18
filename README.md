# Robust-Real-time-Motion-Retargeting-via-Neural-Latent-Prediction
Code for the paper acceptted in 2023IROS "Robust Real-Time Motion Retargeting via Neural Latent Prediction" [here](https://ieeexplore.ieee.org/abstract/document/10342022)

It contains two models: Motion Sequence Retargeting and prediction in latent space

1.Motion Sequence Retargeting:Realize motion sequence retargeting from human motion sequence to robot motion sequence.

2.Prediction in latent space:Predict latent sequence to guide advanced control

## Prerequisite

- [**PyTorch**](https://pytorch.org/get-started/locally/) Tensors and Dynamic neural networks in Python with strong GPU acceleration
- [**pytorch_geometric**](https://github.com/rusty1s/pytorch_geometric) Geometric Deep Learning Extension Library for PyTorch
- [**Kornia**](https://github.com/kornia/kornia) a differentiable computer vision library for PyTorch.
- [**HDF5 for Python**](https://docs.h5py.org/en/stable/) The h5py package is a Pythonic interface to the HDF5 binary data format.

## Dataset

The Chinese sign language dataset can be downloaded [here](https://www.jianguoyun.com/p/DYm5RzMQ74eHChj_lJ0E).


## Citation

If you find this project useful in your research, please cite this paper.

```
@article{wang2023robust,
  title={Robust Real-Time Motion Retargeting via Neural Latent Prediction},
  author={Wang, Tiantian and Zhang, Haodong and Chen, Lu and Wang, Dongqi and Wang, Yue and Xiong, Rong},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3696--3703},
  year={2023},
  organization={IEEE}
}
```
