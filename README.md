# NARL-Adjuster
This is an official PyTorch implementation of Improve Noise Tolerance of Robust Loss via Noise-Awareness.
## Environment
* python 3.7.10
* torch 0.8.1
* torchvision 0.9.1
* sklearn
## Running NARL-Adjuster on benchmark datasets (CIFAR-10 and CIFAR-100).
ResNet32 on CIFAR10 with 20% unif noise:
```
python main.py --dataset cifar10 --corruption_prob 0.2 --corruption_type unif --epochs 120 --warmup_epochs 0 --batch_size 100 --lr 1e-1 --wd 5e-4 --mwd 1e-4
```
ResNet32 on CIFAR100 with 20% unif noise:
```
python main.py --dataset cifar100 --corruption_prob 0.2 --corruption_type unif --epochs 150 --warmup_epochs 0 --batch_size 100 --lr 1e-1 --wd 5e-4 --mwd 1e-4
```
## Results under 40% Nosie (CIFAR10)

| Noise Type | Test Accuracy |
| :----: | :----: |
| Symmetric | 88.10% |
| Asymmetric | 88.03% |
| Instance | 87.33% |
## Acknowledgments
Thanks to the pytorch implementation of Meta-Weight-Net(https://github.com/xjtushujun/meta-weight-net).

Contact: Ding Kehui(dkh19970303@163.com).
