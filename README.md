# NARL-Adjuster
This is an official PyTorch implementation of Improve Noise Tolerance of Robust Loss via Noise-Awareness
## Environment
* python 3.7.10
* torch 0.8.1
* torchvision 0.9.1
## Running this example
ResNet32 on CIFAR10 with 40% uniform noise:
```
python main.py --dataset cifar100 --corruption_prob 0.4 --ctype unif --epochs 150 --warmup_epochs 10 --batch-size 100 --lr 1e-1 --wd 5e-4 -mwd 1e-4
```
