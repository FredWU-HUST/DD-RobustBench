import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate as scipyrotate
import sys

class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(2), img.size(3)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask).to('cuda')
            mask = mask.expand_as(img)
            img *= mask
        return img


class EvaluatorUtils:

    class ParamDiffAug():
        def __init__(self):
            self.aug_mode = 'S' #'multiple or single'
            self.prob_flip = 0.5
            self.ratio_scale = 1.2
            self.ratio_rotate = 15.0
            self.ratio_crop_pad = 0.125
            self.ratio_cutout = 0.5 # the size would be 0.5x0.5
            self.brightness = 1.0
            self.saturation = 2.0
            self.contrast = 0.5

    @staticmethod
    def custom_aug(images, args):
        image_syn_vis = images.clone()
        if args.normalize_data:
            if args.dataset == 'CIFAR10':
                mean = [0.4914, 0.4822, 0.4465]
                std = [0.2023, 0.1994, 0.2010]
            elif args.dataset == 'CIFAR100':
                mean = [0.5071, 0.4866, 0.4409]
                std = [0.2673, 0.2564, 0.2762]
            elif args.dataset == 'tinyimagenet':
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

            for ch in range(3):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
        image_syn_vis[image_syn_vis<0] = 0.0
        image_syn_vis[image_syn_vis>1] = 1.0

        normalized_d = image_syn_vis * 255
        if args.dataset == 'tinyimagenet':
            size = 64
        else:
            size = 32
        if args.aug == 'autoaug':
            if args.dataset == 'tinyimagenet':
                data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET)])
            else:
                data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)])
        elif args.aug == 'randaug':
            data_transforms = transforms.Compose([transforms.RandAugment(num_ops=1)])
        elif args.aug == 'imagenet_aug':
            data_transforms = transforms.Compose([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)])
        elif args.aug == 'cifar_aug':
            data_transforms = transforms.Compose([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()])
        else:
            exit('unknown augmentation method: %s'%args.aug)
        normalized_d = data_transforms(normalized_d.to(torch.uint8))
        normalized_d = normalized_d / 255.0

        # print("changes after autoaug: ", (normalized_d - image_syn_vis).pow(2).sum().item())

        if args.normalize_data:
            for ch in range(3):
                normalized_d[:, ch] = (normalized_d[:, ch] - mean[ch])  / std[ch]

        if args.aug == 'cifar_aug':
            cutout_transform = transforms.Compose([Cutout(16, 1)])
            normalized_d = cutout_transform(normalized_d)

        return normalized_d

    @staticmethod
    def augment(images, dc_aug_param, device):
        if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
            print("using DC augmentation")
            scale = dc_aug_param['scale']
            crop = dc_aug_param['crop']
            rotate = dc_aug_param['rotate']
            noise = dc_aug_param['noise']
            strategy = dc_aug_param['strategy']

            shape = images.shape
            mean = []
            for c in range(shape[1]):
                mean.append(float(torch.mean(images[:,c])))

            def cropfun(i):
                im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
                for c in range(shape[1]):
                    im_[c] = mean[c]
                im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
                r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
                images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

            def scalefun(i):
                h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
                w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
                tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
                mhw = max(h, w, shape[2], shape[3])
                im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
                r = int((mhw - h) / 2)
                c = int((mhw - w) / 2)
                im_[:, r:r + h, c:c + w] = tmp
                r = int((mhw - shape[2]) / 2)
                c = int((mhw - shape[3]) / 2)
                images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

            def rotatefun(i):
                im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
                r = int((im_.shape[-2] - shape[-2]) / 2)
                c = int((im_.shape[-1] - shape[-1]) / 2)
                images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

            def noisefun(i):
                images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


            augs = strategy.split('_')

            for i in range(shape[0]):
                choice = np.random.permutation(augs)[0] # randomly implement one augmentation
                if choice == 'crop':
                    cropfun(i)
                elif choice == 'scale':
                    scalefun(i)
                elif choice == 'rotate':
                    rotatefun(i)
                elif choice == 'noise':
                    noisefun(i)

        return images


    @staticmethod
    def get_daparam(dataset, model, model_eval, ipc):
        # We find that augmentation doesn't always benefit the performance.
        # So we do augmentation for some of the settings.

        dc_aug_param = dict()
        dc_aug_param['crop'] = 4
        dc_aug_param['scale'] = 0.2
        dc_aug_param['rotate'] = 45
        dc_aug_param['noise'] = 0.001
        dc_aug_param['strategy'] = 'none'

        if dataset == 'MNIST':
            dc_aug_param['strategy'] = 'crop_scale_rotate'

        if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
            dc_aug_param['strategy'] = 'crop_noise'

        return dc_aug_param

    @staticmethod
    def pick_gpu_lowest_memory():
        import gpustat
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry['index']), stats)
        ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        return bestGPU



    def set_seed_DiffAug(param):
        if param.latestseed == -1:
            return
        else:
            torch.random.manual_seed(param.latestseed)
            param.latestseed += 1


    def DiffAugment(x, strategy='', seed = -1, param = None):
        AUGMENT_FNS = {
            'color': [EvaluatorUtils.rand_brightness, EvaluatorUtils.rand_saturation, EvaluatorUtils.rand_contrast],
            'crop': [EvaluatorUtils.rand_crop],
            'cutout': [EvaluatorUtils.rand_cutout],
            'flip': [EvaluatorUtils.rand_flip],
            'scale': [EvaluatorUtils.rand_scale],
            'rotate': [EvaluatorUtils.rand_rotate],
        }
        if strategy == 'None' or strategy == 'none' or strategy == '':
            return x

        if seed == -1:
            param.Siamese = False
        else:
            param.Siamese = True

        param.latestseed = seed

        if strategy:
            if param.aug_mode == 'M': # original
                for p in strategy.split('_'):
                    for f in AUGMENT_FNS[p]:
                        x = f(x, param)
            elif param.aug_mode == 'S':
                pbties = strategy.split('_')
                EvaluatorUtils.set_seed_DiffAug(param)
                p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
            else:
                exit('unknown augmentation mode: %s'%param.aug_mode)
            x = x.contiguous()
        return x


    # We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
    def rand_scale(x, param):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = param.ratio_scale
        EvaluatorUtils.set_seed_DiffAug(param)
        sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        EvaluatorUtils.set_seed_DiffAug(param)
        sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if param.Siamese: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape).to(x.device)
        x = F.grid_sample(x, grid)
        return x


    def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
        ratio = param.ratio_rotate
        EvaluatorUtils.set_seed_DiffAug(param)
        theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if param.Siamese: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape).to(x.device)
        x = F.grid_sample(x, grid)
        return x


    def rand_flip(x, param):
        prob = param.prob_flip
        EvaluatorUtils.set_seed_DiffAug(param)
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        if param.Siamese: # Siamese augmentation:
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)


    def rand_brightness(x, param):
        ratio = param.brightness
        EvaluatorUtils.set_seed_DiffAug(param)
        randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if param.Siamese:  # Siamese augmentation:
            randb[:] = randb[0]
        x = x + (randb - 0.5)*ratio
        return x


    def rand_saturation(x, param):
        ratio = param.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        EvaluatorUtils.set_seed_DiffAug(param)
        rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if param.Siamese:  # Siamese augmentation:
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x


    def rand_contrast(x, param):
        ratio = param.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        EvaluatorUtils.set_seed_DiffAug(param)
        randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if param.Siamese:  # Siamese augmentation:
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x


    def rand_crop(x, param):
        # The image is padded on its surrounding and then cropped.
        ratio = param.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        EvaluatorUtils.set_seed_DiffAug(param)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        EvaluatorUtils.set_seed_DiffAug(param)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        if param.Siamese:  # Siamese augmentation:
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x


    def rand_cutout(x, param):
        ratio = param.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        EvaluatorUtils.set_seed_DiffAug(param)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        EvaluatorUtils.set_seed_DiffAug(param)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        if param.Siamese:  # Siamese augmentation:
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x


    @staticmethod
    def compute_std_mean(scores):
        scores = np.array(scores)
        std = np.std(scores)
        mean = np.mean(scores)
        return mean, std

