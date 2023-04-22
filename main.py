# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import random
import numpy as np

from resnet import ResNet32
from resnet import ResNet34
from resnet import VNet
from load_corrupted_data import CIFAR10, CIFAR100


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar10)')
parser.add_argument('--corruption_prob', '--cprob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '--ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip_smi" for cifar10 or "hierarchical" for cifar100).')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='number of epochs for warmup')
parser.add_argument('--batch_size', '--bs', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--meta_weight_decay', '--mwd', default=1e-4, type=float,
                    help='meta weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)


args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = str(0)


print()
print(args)


def build_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_meta_loader, train_loader, test_loader


def build_model():
    if args.corruption_type == 'instance':
        model = ResNet34(args.dataset == 'cifar10' and 10 or 100)
    else:
        model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()

    return model


def gce(outputs, target, q):
    loss = 0
    for i in range(outputs.size(0)):
        x = outputs[i][target[i]]
        if x < 0.003:
            y = 12.9*x
        else:
            y = x ** q[i]
        loss += (1.0 - y) / q[i]

    loss = loss / outputs.size(0)
    return loss.to(device)


def sl(outputs, target, alpha, beta):
    loss = 0
    num_classes = args.dataset == 'cifar10' and 10 or 100
    target = target.type(torch.int64)
    ce = F.cross_entropy(outputs, target.long(), reduce=False)

    pred = F.softmax(outputs, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = torch.nn.functional.one_hot(target, num_classes).float()
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
    loss = (alpha.view(-1,1) * ce.view(-1,1)).mean() + (beta.view(-1,1) * rce.view(-1,1)).mean()

    return loss.to(device)


def poly(outputs, target, lamda, d):
    target = target.type(torch.int64)
    lamda = lamda.view(-1)
    d = d.view(-1)
    celoss = F.cross_entropy(outputs, target, reduce=False)
    
    mask = celoss >= lamda
    loss = torch.FloatTensor((celoss).size()).type_as(celoss)
    loss[mask] = ((d[mask]-1) / d[mask]) * lamda[mask]
    mask = celoss < lamda
    loss[mask] = ((d[mask]-1) / d[mask]) * lamda[mask] * (1-torch.pow(1-celoss[mask]/lamda[mask],(d[mask]/(d[mask]-1))))
    return loss.mean().to(device)


def net_input(outputs,targets):
    targets = targets.type(torch.int64)
#    outputs = F.softmax(outputs,dim=1)
    pred = outputs.gather(1,targets.view(-1,1))
    values, _ = outputs.topk(2, dim=1, largest=True, sorted=True)
    Max, _ = outputs.topk(1, dim=1, largest=True, sorted=True)
    for i in range(outputs.size(0)):
        if values[i,0] == pred[i,0]:
            Max[i,0] = values[i,1]
    diff = pred - Max
    return diff


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epochs):
    if args.corruption_type == 'instance':
        lr = args.lr * ((0.1 ** int(epochs >= 40)) * (0.1 ** int(epochs >= 80)))
    else:
        if args.dataset == 'cifar10':
            lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))
        else:
            lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 120)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def train(train_meta_loader, train_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()        
        inputs, targets   = inputs.to(device), targets.to(device)
        
        if batch_idx % 10 == 0:
            meta_model = build_model()
            meta_model.load_state_dict(model.state_dict())
                
            outputs  = meta_model(inputs)             
            vnet_meta_input  = net_input(outputs, targets)
                    
            outputs  = F.softmax(outputs, dim=1)
            q_meta  = vnet(vnet_meta_input.data.view(-1,1))
            q_meta  = torch.clamp(q_meta, 0.01, 1.)
                
            l_f_meta = gce(outputs, targets, q_meta)
                
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            if args.corruption_type == 'instance':
                meta_lr = args.lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 80)))
            else:
                if args.dataset == 'cifar10':
                    meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
                else:
                    meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120)))
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads
                
            try:
                inputs_val, targets_val = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                inputs_val, targets_val = next(train_meta_loader_iter)
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            y_g_hat = meta_model(inputs_val)
            l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())
            prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]
                
            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()
            meta_loss += l_g_meta.item()

        #gce.q.data = torch.clamp(gce.q.data, 0.01, 1.0).clone().detach()
        outputs  = model(inputs)
        prec_train  = accuracy(outputs.data, targets.data, topk=(1,))[0]        
        vnet_input  = net_input(outputs, targets)

        outputs  = F.softmax(outputs, dim=1)
        with torch.no_grad():
            q  = vnet(vnet_input.view(-1,1))
        q  = torch.clamp(q, 0.01, 1.)
        
        loss = gce(outputs, targets, q)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            # print(q[0:1])
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / ((batch_idx + 1)/10)), prec_train, prec_meta))


def warmup(train_loader, model, optimizer_model, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = F.cross_entropy(outputs, targets.long())

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                      (train_loss / (batch_idx + 1))))


train_meta_loader, train_loader, test_loader = build_dataset()

model = build_model()
vnet = VNet(1, 100, 1).to(device)

optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=args.meta_weight_decay)


def main():
    best_acc = 0
    prec_pic = []

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)

        if epoch < args.warmup_epochs:
            warmup(train_loader, model, optimizer_model, epoch)
        else:
            train(train_meta_loader,train_loader,model,vnet,optimizer_model,optimizer_vnet,epoch)
            
        test_acc = evaluate(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
        prec_pic.append(test_acc)        
        print('best accuracy:', best_acc)

    print('mean',np.mean(prec_pic[args.epochs-5:args.epochs]))
    print('std',np.std(prec_pic[args.epochs-5:args.epochs], ddof=1))

        
if __name__ == '__main__':
    main()
