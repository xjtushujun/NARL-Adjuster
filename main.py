# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt

from load_corrupted_data import CIFAR10, CIFAR100
from resnet import *
from meta import *
from sklearn.cluster import KMeans


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
parser.add_argument('--meta-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.set_defaults(augment=True)


args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


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
        torch.backends.cudnn.benchmark = True

    return model


def gce(outputs, target, q):
    loss = 0
    for i in range(outputs.size(0)):
        x = outputs[i][target[i]]
        if x < 0.003:
            y = 12.9 * x
        else:
            y = x ** q[i]
        loss += (1.0 - y) / q[i]

    loss = loss / outputs.size(0)
    return loss.to(device)
    

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
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy
    
    
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


def train(train_meta_loader, train_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        if batch_idx % args.meta_freq == 0:
            meta_model = build_model().cuda()
            meta_model.load_state_dict(model.state_dict())

            outputs_meta = meta_model(inputs)
            vnet_meta_input = net_input(outputs_meta, targets)

            outputs_meta = F.softmax(outputs_meta, dim=1)
            q_meta = vnet(vnet_meta_input.data.view(-1,1), targets.data, c)
            q_meta = torch.clamp(q_meta, 0.01, 1.)

            l_f_meta = gce(outputs_meta, targets, q_meta)

            meta_grads = torch.autograd.grad(l_f_meta, meta_model.parameters(), create_graph=True)
                
            optimizer_meta = MetaSGD(meta_model, meta_model.parameters(), lr=args.lr)
            optimizer_meta.load_state_dict(optimizer_model.state_dict())
            optimizer_meta.meta_step(meta_grads)

            del meta_grads

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

        outputs = model(inputs)
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]
        vnet_input = net_input(outputs, targets)

        outputs = F.softmax(outputs, dim=1)
        with torch.no_grad():
            q = vnet(vnet_input.view(-1,1), targets.data, c)
        q = torch.clamp(q, 0.01, 1.)

        loss = gce(outputs, targets, q)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
#            print('q', q[0])
#            print('input', vnet_input[0])
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size, (train_loss / (batch_idx + 1)), (meta_loss / ((batch_idx + 1) / 10)), prec_train, prec_meta))


def warmup(train_loader, model, optimizer_model, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = F.cross_entropy(outputs, targets)

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
vnet = Adjuster(1, 100, 100, 1, 3).to(device)

optimizer_model = torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = torch.optim.Adam(vnet.parameters(), 1e-3, weight_decay=args.meta_weight_decay)

a=[]

labels_num = train_loader.dataset.train_labels

for i in range(10):
    a.append([labels_num.count(i)])
    #print(i,' number is ', li.count(i))
print(len(labels_num))

print(a)
es = KMeans(3)
es.fit(a)

c = es.labels_
print(c)
print('c:', c.tolist())

w = [[],[],[]]
for i in range(3):
    for k, j in enumerate(c):
        if i == j:
            w[i].append(a[k][0])

print(w)


def main():
    best_acc = 0

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)

        if epoch < args.warmup_epochs:
            warmup(train_loader, model, optimizer_model, epoch)
        else:
            train(train_meta_loader, train_loader, model, vnet, optimizer_model, optimizer_vnet, epoch)
            
        test_acc = evaluate(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
        print('best accuracy:', best_acc)

        
if __name__ == '__main__':
    main()