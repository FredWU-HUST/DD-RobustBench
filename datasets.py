import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import pickle
from scipy.ndimage.interpolation import rotate as scipyrotate
import kornia as K
from math import ceil
from IPython import embed

class imagenetsubset:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

subset_config = imagenetsubset()

def get_dataset(dataset, data_path, normalize=True,train_batch=128,test_batch=128,zca=False,subset=None,full_transform=False):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        if (not normalize) or zca:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=False, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        if (not normalize) or zca:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=False, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=False, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        if (not normalize) or zca:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=False, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=False, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        # mean = [0.5,0.5,0.5]
        # std = [0.2023, 0.1994, 0.2010]
        std = [0.2471, 0.2435, 0.2616]
        # std = [0.5,0.5,0.5]

        if (not normalize) or zca:
            train_transform = transforms.ToTensor()
            test_transform = transforms.ToTensor()
        elif full_transform:
            train_transform = transforms.Compose([transforms.ColorJitter(),transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # train_transform = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.4, 1), interpolation=InterpolationMode.BILINEAR),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=False, transform=train_transform) # no augmentation
        # dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ColorJitter(),transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])) 
        
        dst_test = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        if (not normalize) or zca:
            train_transform = transforms.ToTensor()
            test_transform = transforms.ToTensor()
        elif full_transform:
            train_transform = transforms.Compose([transforms.ColorJitter(),transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # train_transform = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.4, 1), interpolation=InterpolationMode.BILINEAR),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=False, transform=train_transform) # no augmentation
        # dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transforms.Compose([transforms.ColorJitter(),transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])) 
        
        dst_test = datasets.CIFAR100(data_path, train=False, download=False, transform=test_transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


        ''' 
        If there is an error:

        RuntimeError: CUDA error: device-side assert triggered
        CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
        For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
        Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

        check if you use the ImageNet subsets but don't care about the class dict
        '''

    elif dataset == 'ImageNet':
        channel = 3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if subset==None:
            im_size = (224,224)
        else:
            im_size = (128,128)
        if zca or normalize==False:
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size)])
        if subset==None:
            num_classes = 1000
            dst_train = datasets.ImageNet(data_path, split="train", transform=transform)
            dst_test = datasets.ImageNet(data_path, split="val", transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
            class_names = dst_train.classes
            class_map = {x:x for x in range(num_classes)}
        else:
            num_classes = 10
            subset_config.img_net_classes = subset_config.dict[subset]
            dst_train = datasets.ImageNet(data_path, split="train", transform=transform) # no augmentation
            dst_train_dict = {c : torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.equal(dst_train.targets, subset_config.img_net_classes[c])))) for c in range(len(subset_config.img_net_classes))}
            dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, subset_config.img_net_classes))))
            loader_train_dict = {c : torch.utils.data.DataLoader(dst_train_dict[c], batch_size=train_batch, shuffle=True, num_workers=16) for c in range(len(subset_config.img_net_classes))}
            dst_test = datasets.ImageNet(data_path, split="val", transform=transform)
            dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, subset_config.img_net_classes))))
            for c in range(len(subset_config.img_net_classes)):
                dst_test.dataset.targets[dst_test.dataset.targets == subset_config.img_net_classes[c]] = c
                dst_train.dataset.targets[dst_train.dataset.targets == subset_config.img_net_classes[c]] = c
            # print(dst_test.dataset)
            class_map = {x: i for i, x in enumerate(subset_config.img_net_classes)}
            class_map_inv = {i: x for i, x in enumerate(subset_config.img_net_classes)}
            class_names = None


    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        if full_transform:
            train_transform=transforms.Compose([transforms.ColorJitter(),transforms.RandomHorizontalFlip(),transforms.RandomCrop(64, padding=4)])
        else:
            train_transform=transforms.Compose([transforms.RandomResizedCrop(64)])
        images_train = train_transform(images_train)
        images_train = images_train.detach().float() / 255.0 # totensor
        labels_train = labels_train.detach()
        if normalize and (not zca):
            for c in range(channel):
                images_train[:,c] = (images_train[:,c] - mean[c])/std[c] # normalize
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0 # totensor
        labels_val = labels_val.detach()
        if normalize and (not zca):
            for c in range(channel):
                images_val[:, c] = (images_val[:, c] - mean[c]) / std[c] # normalize

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation
        class_map = {x:x for x in range(num_classes)}

    else:
        exit('unknown dataset: %s'%dataset)

    if zca:
        images = []
        labels = []
        print("Train ZCA")
        for i in range(len(dst_train)):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)

        images = []
        labels = []
        print("Test ZCA")
        for i in range(len(dst_test)):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)


    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=train_batch, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=test_batch, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, trainloader,testloader,class_map


def load_distilled_dataset(method, pt_path,dataset='CIFAR10',factor=2):
    methods = ['DC','DSA','DD','DM','MTT','SRe2L','TESLA','IDM','DREAM','IDC']
    assert method in methods,'Distillation method not recognized.\nAvailable methods:%s'%(methods)
    if method == 'DC' or method == 'DSA' or method == 'DM':
        data = torch.load(pt_path)
        data = data['data']
        train_images, train_labels = data[-1]
    elif method == 'DD':
        data = torch.load(pt_path)
        train_images, train_labels = data[-1][0],data[-1][1]
    elif method == 'MTT' or method == 'TESLA':
        assert os.path.isdir(pt_path),'MTT pt files should be in a folder.'
        if len(os.listdir(pt_path)) == 2:
            train_images = torch.load(os.path.join(pt_path,'images_best.pt'))
            train_labels = torch.load(os.path.join(pt_path,'labels_best.pt')).long()
        else: # if tinyimagenet with ipc=50, the data are separated to 5 files.
            images = []
            labels = []
            for i in range(5):
                images.append(torch.load(os.path.join(pt_path,'images_best_'+str(i)+'.pt')))
                labels.append(torch.load(os.path.join(pt_path,'labels_best_'+str(i)+'.pt')).long())
            train_images = torch.cat(images)
            train_labels = torch.cat(labels)
    elif method == 'SRe2L':
        train_transform = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.4, 1), interpolation=InterpolationMode.BILINEAR),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])])
        dataset = datasets.ImageFolder(pt_path,transform=train_transform)
        train_images = []
        train_labels = []
        for img, target in dataset:
            train_images.append(img)
            train_labels.append(target)
        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels)
    elif method=='IDM':
        with open(pt_path, 'rb') as f:
            loaded_data = pickle.load(f)
        train_images, train_labels = loaded_data
        train_images, train_labels = number_sign_augment(train_images, train_labels)
    elif method=='DREAM' or method=='IDC':
        data_dec = []
        target_dec = []
        if dataset=='CIFAR10':
            ncls,imsize=10,(32,32)
            mean=[0.4914, 0.4822, 0.4465]
            std=[0.2471, 0.2435, 0.2616]
        elif dataset=='CIFAR100':
            ncls,imsize=100,(32,32)
            mean=[0.5071, 0.4866, 0.4409]
            std=[0.2673, 0.2564, 0.2762]
        elif dataset=='TinyImageNet':
            ncls,imsize=200,(64,64)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        data,target=torch.load(pt_path)
        ipc = int(data.shape[0]//ncls)
        for c in range(ncls):
            idx_from = ipc * c 
            idx_to = ipc * (c + 1)
            d = data[idx_from:idx_to].detach()
            t = target[idx_from:idx_to].detach()
            d,t = dream_decode(d, t,imsize,factor)
            data_dec.append(d)
            target_dec.append(t)
        data_dec=torch.cat(data_dec)
        target_dec = torch.cat(target_dec)
        for c in range(3):
            data_dec[:,c] = (data_dec[:,c] - mean[c])/std[c] # normalize
        train_images,train_labels = data_dec,target_dec

    return train_images, train_labels


# for idm
def number_sign_augment(image_syn, label_syn):
    half_length = image_syn.shape[2]//2
    # import pdb; pdb.set_trace()
    a, b, c, d = image_syn[:, :, :half_length, :half_length].clone(), image_syn[:, :, half_length:, :half_length].clone(), image_syn[:, :, :half_length, half_length:].clone(), image_syn[:, :, half_length:, half_length:].clone()
    a, b, c, d = F.upsample(a, scale_factor=2, mode='bilinear'), F.upsample(b, scale_factor=2, mode='bilinear'), \
        F.upsample(c, scale_factor=2, mode='bilinear'), F.upsample(d, scale_factor=2, mode='bilinear')
    # a, b, c, d = image_syn.clone(), image_syn.clone(), image_syn.clone(), image_syn.clone()
    image_syn_augmented = torch.concat([a, b, c, d], dim=0)
    label_syn_augmented = label_syn.repeat(4)
    return image_syn_augmented, label_syn_augmented

# for dream
def dream_decode(data, target,imsize,factor,bound=128,decode_type='single'):
    """Multi-formation
    """
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, imsize,factor)
        elif decode_type == 'bound':
            data, target = decode_zoom_bound(data, target, imsize,factor, bound=bound)
        else:
            data, target = decode_zoom(data, target, imsize,factor)

    return data, target


def decode_zoom(img, target, imsize,factor):
    """Uniform multi-formation
    """
    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    resize = nn.Upsample(size=imsize, mode='bilinear')
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec

def decode_zoom_multi(img, target, imsize,factor_max):
    """Multi-scale multi-formation
    """
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, imsize,factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)

def decode_zoom_bound(img, target, imsize,factor_max, bound=128):
    """Uniform multi-formation with bounded number of synthetic data
    """
    bound_cur = bound - len(img)
    budget = len(img)

    data_multi = []
    target_multi = []

    idx = 0
    decoded_total = 0
    for factor in range(factor_max, 0, -1):
        decode_size = factor**2
        if factor > 1:
            n = min(bound_cur // decode_size, budget)
        else:
            n = budget

        decoded = decode_zoom(img[idx:idx + n], target[idx:idx + n], imsize,factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

        idx += n
        budget -= n
        decoded_total += n * decode_size
        bound_cur = bound - decoded_total - budget

        if budget == 0:
            break

    data_multi = torch.cat(data_multi)
    target_multi = torch.cat(target_multi)
    return data_multi, target_multi

def decode(self, data, target, bound=128):
    """Multi-formation
    """
    if self.factor > 1:
        if self.decode_type == 'multi':
            data, target = self.decode_zoom_multi(data, target, self.factor)
        elif self.decode_type == 'bound':
            data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
        else:
            data, target = self.decode_zoom(data, target, self.factor)

    return data, target



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]

if __name__ == '__main__':
    im,la = load_distilled_dataset('DREAM',"./best.pt")
    embed(h='end')