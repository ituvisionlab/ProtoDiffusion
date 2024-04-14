from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10, ImageFolder
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from typing import Tuple

def load_cifar10(batchsize:int, numworkers:int) -> Tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root='./datasets/cifar10',
                        train=True,
                        download=True,
                        transform=trans
                    )
    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        sampler=sampler,
                        drop_last=True
                    )
    return trainloader, sampler

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5


def load_stl10(batchsize:int, numworkers:int, image_size: int) -> Tuple[DataLoader, DistributedSampler]:
    if image_size == -1:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
    else:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size,image_size)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        
    data_train = STL10(
                        root='./datasets/stl10',
                        split='train',
                        download=True,
                        transform=trans
                        )
    data_test = STL10(
                        root='./',
                        split='test',
                        download=True,
                        transform=trans
                        )
    # merge train and test
    data_train.data = np.concatenate([data_train.data,data_test.data],axis=0)
    data_train.labels = np.concatenate([data_train.labels,data_test.labels],axis=0)

    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        sampler=sampler,
                        drop_last=True
                    )
    return trainloader,sampler


def load_tiny_imagenet(batchsize:int, numworkers:int, image_size: int, image_dir: str) -> Tuple[DataLoader, DistributedSampler]:
    if image_size == -1:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
    else:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size,image_size)),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])

    data_train = ImageFolder(image_dir, transform=trans)
    # train data path, *./datasets/tiny_imagenet/train

    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        sampler=sampler,
                        drop_last=True
                    )
    return trainloader,sampler