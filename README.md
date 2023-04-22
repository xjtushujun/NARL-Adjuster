# NARL-Adjuster
This is an official PyTorch implementation of Improve Noise Tolerance of Robust Loss via Noise-Awareness
## Environment
* python 3.7.10
* torch 0.8.1
* torchvision 0.9.1
## Running this example
ResNet32 on CIFAR10 with 40% unif noise:
```
python main.py --dataset cifar10 --corruption_prob 0.4 --corruption_type flip_smi --epochs 120 --warmup_epochs 0 --batch-size 100 --lr 1e-1 --wd 5e-4 -mwd 1e-4
```
## Result
