import torch
import torch.nn.functional as F
import numpy as np


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


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


def sl(outputs, target, alpha, beta):
    loss = 0
    num_classes = int(outputs.size(1))
    target = target.type(torch.int64)
    ce = F.cross_entropy(outputs, target.long(), reduce=False)

    pred = F.softmax(outputs, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = torch.nn.functional.one_hot(target, num_classes).float()
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = (-1*torch.sum(pred * torch.log10(label_one_hot), dim=1))
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


def custom_kl_div(prediction, target, w):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    output = w * output.view(-1,1)
    return output.mean()


class JensenShannonDivergenceWeightedScaled(torch.nn.Module):
    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedScaled, self).__init__()
        self.num_classes = num_classes
        self.weights =  [float(w) for w in weights]
        
        self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001
    
    def forward(self, pred, labels):
#        print(self.weights[0].size())
        self.scale = -1.0 / ((1.0-self.weights[0]) * torch.log((1.0-self.weights[0])))
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1)) 
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float() 
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        jsw = sum([custom_kl_div(mean_distrib_log, d, w * self.scale) for w,d in zip(self.weights, distribs)])
        return jsw

JS = JensenShannonDivergenceWeightedScaled(num_classes=10,weights=[0.7,0.3])

def loss_forward(outputs,targets,vnet_input,vnet,c,loss_used='GCE'):
    if loss_used == 'GCE':
        outputs = F.softmax(outputs, dim=1)
        q = vnet(vnet_input.data.view(-1,1), targets.data, c)
        q = torch.clamp(q, 0.01, 1.)
        loss = gce(outputs, targets, q)
    elif loss_used == 'SL':
        a = vnet(vnet_input.data.view(-1,1), targets.data, c)[:,0]
        b = vnet(vnet_input.data.view(-1,1), targets.data, c)[:,1]
        a = torch.clamp(a, 0., 20.)
        b = torch.clamp(b, 0., 20.)
        loss = sl(outputs, targets, a, b)
    elif loss_used == 'Poly':
        a = vnet(vnet_input.data.view(-1,1), targets.data, c)[:,0]
        b = vnet(vnet_input.data.view(-1,1), targets.data, c)[:,1]
        a = torch.clamp(a, 0., 20.)
        b = torch.clamp(b, 1.01, 20.)
        loss = sl(outputs, targets, a, b)
    elif loss_used == 'JS':
        q = vnet(vnet_input.data.view(-1,1), targets.data, c)
        q = torch.clamp(q, 0.01, 0.99)
        JS.weights = list([q, 1-q])
        loss = JS(outputs, targets)
    return loss