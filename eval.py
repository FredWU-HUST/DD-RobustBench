'''
Evaluate trained models on test set.
'''
import argparse
import torch
import torchvision
from torchvision import transforms
import torchvision.models as models
import os
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0,'.')
from IPython import  embed
from datasets import get_dataset,TensorDataset
from models import *



def evalTestAcc(net,testDataLoader,class_num,device,class_dict):
    net.eval()
    totalAcc = 0.0
    total = 0.0
    with torch.no_grad():
        for idx, dataset in enumerate(testDataLoader):
            data, labelOrg = dataset
            labelOrg = [class_dict[x] for x in labelOrg.tolist()]
            labelOrg = torch.tensor(labelOrg, dtype=torch.long).to(device)
            predict = net(data.to(device))
            _, predicted = torch.max(predict.data, dim=1)
            totalAcc += predicted.cpu().eq(labelOrg.cpu()).sum()
            label = F.one_hot(labelOrg.to(torch.long), class_num).to(torch.float).to(device)
            total += label.size(0)
    return totalAcc / total


def eval():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default=None, help='imagenet subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--pt_path', type=str, default="./trained_models/cifar10_best.pt", help='weights.pt path')
    parser.add_argument('--eval_batch', type=int, default=256, help='eval batch size')
    parser.add_argument('--zca', action="store_true", default=False, help='zca')
    parser.add_argument('--norm', action="store_false",default=True, help='normalize')
    parser.add_argument('--whole_model', action='store_true', default=False, help='loaded model contains full information')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    print('Eval model %s on dataset %s.'%(args.model,args.dataset))
    print('Weights loaded from %s'%(args.pt_path))

    channel, im_size, num_classes, _, _, _, dst_train, dst_test, trainloader, testloader,classdict = get_dataset(args.dataset, args.data_path,normalize=args.norm,zca=args.zca,subset=args.subset)

    ## Customized dataset if you want to load your own dataset
    # im = torch.load("./CIFAR10/my_images.pt")
    # la = torch.load("./CIFAR10/my_labels.pt").cpu().int()
    # dst_test = TensorDataset(im,la)
    
    test_dataLoader = DataLoader(dst_test,batch_size=args.eval_batch,shuffle=False)
    args.class_num = num_classes

    model = get_network(args.model,channel,num_classes,im_size,parallel=True).to(args.device)

    weights = torch.load(args.pt_path, map_location=args.device)
    # For different saving formats of weight files
    if args.whole_model:
        if 'state_dict' in weights.keys():
            weights = weights['state_dict']
        elif 'model' in weights.keys():
            weights = weights['model']
    model.load_state_dict(weights)
    model = nn.DataParallel(model).cuda()
    eval_acc = evalTestAcc(model,test_dataLoader,args.class_num,args.device,classdict)

    print('Acc = %.3f%%'%(eval_acc*100))


# CUDA_VISIBLE_DEVICES=0 python eval.py --dataset CIFAR10 --model ConvNet --data_path ./data/ --pt_path ./trained_models/cifar10_best.pt
if __name__ == '__main__':
    eval()