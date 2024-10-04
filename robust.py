'''
Perturbing original datasets and test robust accuracy.
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
# import torchattacks
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
from datasets import get_dataset
from models import get_network
import yaml
import logging
from attack_utils import *

from IPython import embed



def evalRobustAcc(net,testDataLoader,attacker,class_num,device,class_dict):
    net.eval()
    totalAcc = 0.0
    total = 0.0
    for dataset in tqdm(testDataLoader,leave=False):
        data, labelOrg = dataset
        labelOrg = [class_dict[x] for x in labelOrg.tolist()]
        labelOrg = torch.tensor(labelOrg, dtype=torch.long).to(device)
        # perturb clean images
        data = attacker.perturb(data,labelOrg)
        predict = net(data.to(device))
        _, predicted = torch.max(predict.data, dim=1)
        totalAcc += predicted.cpu().eq(labelOrg.cpu()).sum()
        label = F.one_hot(labelOrg.to(torch.long), class_num).to(torch.float).to(device)
        total += label.size(0)
    return totalAcc / total


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--params', type=str, default='./configs/params.yaml', help='path to yaml flie which save parameters for attacks')
    parser.add_argument('--weights', type=str, default='./configs/weights.yaml', help='path to yaml flie which save weight file path')
    parser.add_argument('--attacker', type=str, default='VANILA', help='attack algorithm')
    parser.add_argument('--repeat', type=int, default=1, help='repeat exp and calculate average robust acc')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--data_path', type=str, default='./data/', help='dataset path')
    parser.add_argument('--normalize_data', action="store_false", default=True, help='if models trained on normalized dataset')
    parser.add_argument('--subset', type=str, default=None, help='for imagenet-subset, for example imagenette')
    parser.add_argument('--whole_model', action='store_true', default=False, help='loaded model contains full information')
    parser.add_argument('--log_path', type=str, default='./attacking_results/attack_results.log', help='save robust accuracy in log file')


    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.attacker = args.attacker.upper()

    logging.basicConfig(filename=args.log_path, filemode="w", format="%(asctime)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    

    print('Using %s for perturbation.'%args.attacker)
    logging.info('Using %s for perturbation.'%args.attacker)


    channel, im_size, num_classes, _, mean,std, dst_train, dst_test, trainloader, testloader,classdict = get_dataset(args.dataset,args.data_path,args.normalize_data,test_batch=args.batch_size,subset=args.subset)
    
    print('Repeat each experiment for %d times.'%args.repeat)
    logging.info('Repeat each experiment for %d times.'%args.repeat)

    # Load parameters from yaml file
    with open(args.params, 'r') as file:
        config = yaml.safe_load(file)
    params = config[args.attacker]

    # Load weight path from yaml file
    with open(args.weights, 'r') as file:
        config = yaml.safe_load(file)
    paths = config['Path']
    
    for param in params:
        print('Test parameters: %s'%(str(param)))
        logging.info('Test parameters: %s'%(str(param)))
        # AutoAttack need class number
        if args.attacker == 'AUTOATTACK':
            param.append(num_classes)
        for path in paths:
            basename = os.path.basename(path)
            print('Testing weights: %s'%basename)
            logging.info('Testing weights: %s'%basename)
            model = get_network(args.model,channel,num_classes,im_size,parallel=True).to(device)
            weights = torch.load(path, map_location=device)
            if args.whole_model:
                weights = weights['state_dict']
            model.load_state_dict(weights)
            model = nn.DataParallel(model).cuda()
            model.eval()
            acc = []
            for _ in range(args.repeat):
                attacker = get_attacker(args.attacker)
                attacker = attacker(model,params=param,normalize=args.normalize_data,mean=mean,std=std)
                robustacc = evalRobustAcc(model,testloader,attacker,num_classes,device,classdict)
                acc.append(robustacc)
            acc = torch.tensor(acc)
            print('Average robust accuracy: %.2f%%'%(100*acc.mean()))
            logging.info('Average robust accuracy: %.2f%%'%(100*acc.mean()))





# CUDA_VISIBLE_DEVICES=0 python robust.py --dataset CIFAR10 --model ConvNet --attacker CW 
if __name__=='__main__':
    main()