r'''
Train networks from scratch using distilled datasets or original datasets.
'''

import argparse
import torch
import torchvision
import os
from datasets import get_dataset
from models import *
from datasets import *
from augmentations import EvaluatorUtils
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as mm
import time
import numpy as np
from datetime import timedelta
from IPython import  embed
import copy

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time   
    return timedelta(seconds=int(round(time_dif)))

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def train_and_eval(net,trainloader,testloader,optimizer,criterion,class_dict,args):
    net = net.to(args.device)
    lr = float(args.lr)
    Epoch = int(args.train_epoch)
    # customized learning rate scheduler
    def lr_lambda(epoch):
        if epoch < 500:
            return 1
        elif epoch < 1000:
            return 0.1
        else:
            return 0.01
    lr_schedule = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    start = time.time()
    best_test_acc = 0
    for ep in range(Epoch+1):
        _,loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion,class_dict, args, aug = True, ep=ep)
        # update scheduler
        lr_schedule.step()
        # record training time
        time_train = time.time() - start
        # eval on val set 
        if ep%args.eval_gap == 0 or ep==Epoch:
            print('Current best test acc %.4f%%'%(100*best_test_acc))
            model,_, acc_test = epoch('test', testloader, net, optimizer, criterion,class_dict, args, aug = False, ep=0)
            print('%s Evaluate: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f%%, test acc = %.4f%%' % (get_time(), ep, int(time_train), loss_train, acc_train*100, acc_test*100))
        # save current best checkpoint
        if best_test_acc<acc_test:
            best_test_acc = acc_test
            ckpt_path = os.path.join(args.save_path,'best.pt')
            torch.save(model.state_dict(),ckpt_path)
            print('save current best checkpoint to %s'%(ckpt_path))
        # save checkpoint every save_gap epochs
        if ep % args.save_gap == 0 and ep != 0: 
            ckpt_path = os.path.join(args.save_path,'checkpoints_ep%d_testacc%.2f.pt'%(ep,acc_test*100))
            torch.save(model.state_dict(),ckpt_path)
            print('save checkpoint to %s'%(ckpt_path))
    return net,acc_train, acc_test, best_test_acc

def epoch(mode, dataloader, net, optimizer, criterion,class_dict, args, aug, ep):
    # print(optimizer.param_groups[0]['lr'])
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)
    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                if i_batch == 0 and mode == 'train':
                    print("using dsa")
                img = EvaluatorUtils.DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            elif hasattr(args, 'aug') and args.aug != '' and mode == 'train':
                if i_batch == 0:
                    print("using ", args.aug)
                img = EvaluatorUtils.custom_aug(img, args)
            else:
                if i_batch == 0:
                    if args.dc_aug_param == None or args.dc_aug_param['strategy'] == 'none':
                        print("not using any augmentations")
                img = EvaluatorUtils.augment(img, args.dc_aug_param, device=args.device)

        lab_all = [class_dict[x] for x in datum[1].tolist()]
        lab = torch.tensor(lab_all, dtype=torch.long).to(args.device)
        n_b = lab.shape[0]
        output = net(img)
        loss = criterion(output, lab)
        if lab.dtype == torch.float:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=1)))
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp
    model = copy.deepcopy(net)

    return model,loss_avg, acc_avg



def train():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--distilled', action="store_true", default=False, help='is distilled data or not')
    parser.add_argument('--distill_method', type=str, default='DC', help='distill algorithm')
    parser.add_argument('--ipc', type=int, default=1, help='distill ipc')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--model_num', type=int, default=1, help='how many models to train')
    parser.add_argument('--train_batch', type=int, default=256, help='train batch size')
    parser.add_argument('--eval_batch', type=int, default=256, help='eval batch size')
    parser.add_argument('--train_epoch', type=int, default=1000, help='train epochs')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--pt_path', type=str, default='./data', help='distilled data pt file path')
    parser.add_argument('--save_path', type=str, default='./training_results/', help='path to save results')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--save_gap', type=int, default=50, help='how often to save a checkpoint')
    parser.add_argument('--eval_gap', type=int, default=1, help='how many iters to eval')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer used in training')
    parser.add_argument('--dsa', action="store_true", default=False, help='dsa')
    parser.add_argument('--aug', type=str, default='', help='augmentation method')
    parser.add_argument('--normalize_data', action="store_true", default=True, help='the number of evaluating randomly initialized models')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--zca', action="store_true", default=False, help='zca')
    parser.add_argument('--pretrained_path', type=str, default=None, help='path to load pretrained wieght')
    parser.add_argument('--subset', type=str, default=None, help='for imagenet-subset, for example imagenette')
    parser.add_argument('--full_transform', action="store_true", default=False, help='use full transforms (colorjittering etc.) when loading original train set')
    parser.add_argument('--factor', type=int, default=2, help='muti-formation factor')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.seed != None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
    if args.dsa:
        args.dsa_param = EvaluatorUtils.ParamDiffAug()
        args.dc_aug_param = None
    if args.aug != '':
        args.dc_aug_param = None
    if args.dc_aug_param != None and args.dc_aug_param['strategy'] != 'none':
        pass
    
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    

    print('Train model %s for %d epochs, with batch size %d, on dataset %s (distilled:%s).'%(args.model,args.train_epoch,args.train_batch,args.dataset,args.distilled))
    print('Save results to %s'%(args.save_path))

    if args.dataset=='ImageNet' and args.subset!=None:
        print('Using ImageNet subset: '+args.subset)

    channel, im_size, num_classes, _, _, _, dst_train, dst_test, trainloader, testloader, class_dict = get_dataset(args.dataset, args.data_path,zca=args.zca,subset=args.subset,full_transform=args.full_transform)
    
    if args.distilled: # load distilled dataset for training
        train_images,train_labels = load_distilled_dataset(args.distill_method,args.pt_path,args.dataset,args.factor)
        dst_train = TensorDataset(train_images, train_labels)
    
    train_dataLoader = DataLoader(dst_train,batch_size=args.train_batch,shuffle=False)
    test_dataLoader = DataLoader(dst_test,batch_size=args.eval_batch,shuffle=False)
    
    args.class_num = num_classes
    save_dir = args.save_path

    total_acc = np.array([])
    for exp in range(args.model_num):
        print('\n================== Model %d ==================\n '%exp)
        model = get_network(args.model,channel,num_classes,im_size).to(args.device)
        if args.pretrained_path != None:
            ckpt = torch.load(args.pretrained_path)
            model.load_state_dict(ckpt)

        train_data = train_dataLoader
        eval_data = test_dataLoader
        if args.optimizer == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            # optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
            # optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
        optim.zero_grad()
        lossfunc = nn.CrossEntropyLoss().to(args.device)
        args.save_path = os.path.join(save_dir,'model_'+str(exp))
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        model, acc_train, acc_test, best_test_acc = train_and_eval(model,train_data,eval_data,optim,lossfunc,class_dict,args)
        total_acc = np.append(total_acc,best_test_acc)
        torch.save(model.state_dict(), os.path.join(args.save_path,'latest.pt'))

    ave_acc = np.mean(total_acc)  
    best_acc = np.amax(total_acc)
    best_acc_index = np.argmax(total_acc)

    print('Average acc on test set = %.3f%%'%(ave_acc*100))
    print('Best acc = %.3f%%, in model %d'%(best_acc*100,best_acc_index))
    logs = 'Accs = %s, average acc = %.4f%%, best acc = %.4f%% in model_%d'%(total_acc,ave_acc*100,best_acc*100,best_acc_index)
    with open(os.path.join(save_dir,'results.txt'), 'w') as fw:
        fw.write(logs)
    print('Save results to %s'%(save_dir))


# train on original dataset
# CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset CIFAR10 --model ConvNet --model_num 5 --train_batch 1024  --train_epoch 2000 --save_path ./result/convnet_cifar10 --optimizer sgd  --full_transform
# train on distilled dataset
# CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset CIFAR10 --model ConvNet --model_num 5 --train_batch 256  --train_epoch 1000 --save_path ./result/convnet_cifar10_dc --optimizer sgd --distilled --distill_method xxx --pt_path xxx (--dsa)
if __name__ == '__main__':
        train()