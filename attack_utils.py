'''
Attacking algorithms 
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchattacks
from torchvision import transforms
import sys
import copy
from Auto import AutoAttack

from IPython import embed


ATTACKERS = ['FGSM','PGD','CW','VMIFGSM','JITTER','VANILA','AUTOATTACK']

def get_attacker(name):
    name = name.upper()
    assert name in ATTACKERS,'Attack not available!'
    if name == 'VMIFGSM':
        return ATTACK_VMIFGSM
    elif name== 'VANILA':
        return ATTACK_VANILA
    elif name== 'AUTOATTACK':
        return ATTACK_AUTO
    elif name== 'CW':
        return ATTACK_CW
    elif name== 'PGD':
        return ATTACK_PGD
    elif name== 'FGSM':
        return ATTACK_FGSM
    elif name== 'JITTER':
        return ATTACK_Jitter
    


def Normalize(X,mu,std):
    if type(mu)==list:
        mu = torch.tensor(mu).view(3, 1, 1).cuda()
    if type(std)==list:
        std = torch.tensor(std).view(3, 1, 1).cuda()
    return (X - mu)/std

def de_Normalize(X,mu,std):
    if type(mu)==list:
        mu = torch.tensor(mu).view(3, 1, 1).cuda()
    if type(std)==list:
        std = torch.tensor(std).view(3, 1, 1).cuda()
    return X*std+mu



'''
Pack up attacks.
Input: model, parameters, clean images, labels
Output: perturbed images
'''
class ATTACKS:
    def __init__(self, model, params=None, normalize=True, mean=None, std=None):
        self.model = copy.deepcopy(model).eval()
        self.params = params
        self.normalize = normalize
        self.mean = mean
        self.std =std
        if normalize:
            assert mean!=None and std!=None, 'Need mean and std for attackers!'


class ATTACK_AUTO(ATTACKS):
    def __init__(self, model, params=[1/255,'standard',10], normalize=True, mean=None, std=None):
        super().__init__(model, params, normalize, mean, std)
        assert len(params)==3, 'Parameters for AutoAttack invalid!'
        # eps, standard/plus/rand/apgd-ce/apgd-t/fab-t/square, class number
        self.attacker = AutoAttack(model,eps=params[0],version=params[1],n_classes=params[2])
        if normalize:
            self.attacker.set_normalization_used(mean=self.mean,std=self.std)
    
    def perturb(self,images,labels):
        adv_images = self.attacker(images,labels)
        return adv_images


class ATTACK_VANILA(ATTACKS):
    def __init__(self, model, params=None, normalize=True, mean=None, std=None):
        super().__init__(model, params, normalize, mean, std)
        self.attacker = torchattacks.VANILA(model)
        if normalize:
            self.attacker.set_normalization_used(mean=self.mean,std=self.std)
    
    def perturb(self,images,labels):
        adv_images = self.attacker(images,labels)
        return adv_images

class ATTACK_FGSM(ATTACKS):
    def __init__(self, model, params=[2/255], normalize=True, mean=None, std=None):
        super().__init__(model, params, normalize, mean, std)
        # eps
        assert len(params)==1, 'Parameters for FGSM invalid!'
        self.attacker = torchattacks.FGSM(model,eps=self.params[0])
        if normalize:
            self.attacker.set_normalization_used(mean=self.mean,std=self.std)
    
    def perturb(self,images,labels):
        adv_images = self.attacker(images,labels)
        return adv_images
    
class ATTACK_PGD(ATTACKS):
    def __init__(self, model, params=[2/255,2/255,10], normalize=True, mean=None, std=None):
        super().__init__(model, params, normalize, mean, std)
        # eps, alpha, steps
        assert len(params)==3, 'Parameters for PGD invalid!'
        self.attacker = torchattacks.PGD(model,eps=self.params[0],alpha=self.params[1],steps=self.params[2])
        if normalize:
            self.attacker.set_normalization_used(mean=self.mean,std=self.std)
    
    def perturb(self,images,labels):
        adv_images = self.attacker(images,labels)
        return adv_images

class ATTACK_VMIFGSM(ATTACKS):
    def __init__(self, model, params=[2/255,2/255,10], normalize=True, mean=None, std=None):
        super().__init__(model, params, normalize, mean, std)
        # eps, alpha, steps
        assert len(params)==3, 'Parameters for VMIFGSM invalid!'
        self.attacker = torchattacks.VMIFGSM(model,params[0],params[1],params[2])
        if normalize:
            self.attacker.set_normalization_used(mean=self.mean,std=self.std)
    
    def perturb(self,images,labels):
        adv_images = self.attacker(images,labels)
        return adv_images
    
class ATTACK_Jitter(ATTACKS):
    def __init__(self, model, params=[2/255,2/255,10], normalize=True, mean=None, std=None):
        super().__init__(model, params, normalize, mean, std)
        # eps, alpha, steps
        assert len(params)==3, 'Parameters for Jitter invalid!'
        self.attacker = torchattacks.Jitter(model,params[0],params[1],params[2])
        if normalize:
            self.attacker.set_normalization_used(mean=self.mean,std=self.std)
    
    def perturb(self,images,labels):
        adv_images = self.attacker(images,labels)
        return adv_images

class ATTACK_CW(ATTACKS):
    def __init__(self, model, params=[2/255,2/255,10], normalize=True, mean=None, std=None, norm='l_inf',early_stop=False,random_start=True):
        super().__init__(model, params, normalize, mean, std)
        self.norm = norm
        self.early_stop = early_stop
        assert len(params)==3, 'Parameters for CW invalid!'
        self.epsilon = params[0]
        self.alpha = params[1]
        self.attack_iters = params[2]
        self.random_start = random_start
        # self.mean = torch.tensor(self.mean).view(3, 1, 1).cuda()
        # self.std = torch.tensor(self.std).view(3, 1, 1).cuda()
    
    
    def clamp(self,X, lower_limit, upper_limit):
        lower_limit = torch.tensor(lower_limit,device=X.device)
        upper_limit = torch.tensor(upper_limit,device=X.device)
        return torch.max(torch.min(X, upper_limit), lower_limit)
    
    def one_hot_tensor(self,y_batch_tensor, num_classes):
        y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0)
        y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
        return y_tensor
    
    def CW_loss(self, logits, targets, margin = 50., reduce = False):
        n_class = logits.size(1)
        onehot_targets = self.one_hot_tensor(targets, n_class)
        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]
        loss = -torch.sum(torch.clamp(self_loss - other_loss + margin, 0))
        if reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num
        return loss
    
    def get_delta(self,net, X, y, norm='l_inf', early_stop=False):
        delta = torch.zeros_like(X).cuda()
        if self.random_start:
            if norm == "l_inf":
                delta.uniform_(-self.epsilon, self.epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*self.epsilon
            else:
                raise ValueError
        delta = self.clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(self.attack_iters):
            output = net(Normalize(X + delta,self.mean,self.std))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = self.CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + self.alpha * torch.sign(g), min=-self.epsilon, max=self.epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*self.alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=self.epsilon).view_as(d)
            d = self.clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        return delta.detach()

    def perturb(self,images,labels):
        images = images.cuda()
        if self.normalize:
            images = de_Normalize(images,self.mean,self.std)
        delta = self.get_delta(self.model,images,labels,norm=self.norm,early_stop=self.early_stop)
        delta = delta.detach()
        adv_images = Normalize(torch.clamp(images + delta[:images.size(0)], min=0, max=1),mu=self.mean,std=self.std)
        return adv_images

