# NARL-Adjuster
This is an official PyTorch implementation for the paper: Improve Noise Tolerance of Robust Loss via Noise-Awareness. [Arxiv Vervision](https://arxiv.org/pdf/2301.07306)

## Overview
Robust loss minimization is an important strategy for handling robust learning issue on noisy labels. Current approaches for designing robust losses involve the introduction of noise-robust factors, i.e., hyperparameters, to control the trade-off between noise robustness and learnability. However, finding suitable hyperparameters for different datasets with noisy labels is a challenging and time-consuming task. Moreover, existing robust loss methods usually assume that all training samples share common hyperparameters, which are independent of instances. This limits the ability of these methods to distinguish the individual noise properties of different samples and overlooks the varying contributions of diverse training samples in helping models understand underlying patterns. To address above issues, we propose to assemble robust loss with instance-dependent hyperparameters to improve their noise tolerance with theoretical guarantee. To achieve setting such instance-dependent hyperparameters for robust loss, we propose a meta-learning method which is capable of adaptively learning a hyperparameter prediction function, called Noise-Aware-Robust-Loss-Adjuster (NARL-Adjuster for brevity). The structure of NARL-Adjuster is shown below:

![image](Architecture.pdf)

## Environment
* python 3.7.10
* torch 0.8.1
* torchvision 0.9.1
* sklearn

## Experiments
We empirically validate that the proposed NARL-Adjuster algorithm can enhance the robustness of loss functions on four kinds of robust losses (GCE, SL, JS and PolySoft) and conduct experiments on benchmark datasets (CIFAR-10 and CIFAR-100) under different nosie structures (uniform, class-dependent and instance-dependent). Here are examples for GCE:

CIFAR10 with 40% class-dependent noise:
```
python main.py --dataset cifar10 --corruption_prob 0.2 --corruption_type flip_smi --loss_type GCE --epochs 120 --batch_size 100 --lr 1e-1 --wd 5e-4 --mwd 1e-4
```
CIFAR100 with 40% class-dependent noise:
```
python main.py --dataset cifar100 --corruption_prob 0.2 --corruption_type hierarchical --loss_type GCE --epochs 150 --batch_size 100 --lr 1e-1 --wd 5e-4 --mwd 1e-4
```
## Acknowledgments
Thanks to the pytorch implementation of Meta-Weight-Net(https://github.com/xjtushujun/meta-weight-net).
Contact: Ding Kehui(dkh19970303@163.com).
