
import torch
import numpy as np
from torchvision import datasets, transforms


# # https://github.com/aaron-xichen/pytorch-playground/blob/master/svhn/train.py
# def target_transform(target):
#         return int(target[0]) - 1

data_path = "./data"
batch_size = None

def load_svhn_help():
    train_dataset = datasets.SVHN(
                    root=data_path, split='train', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                    #target_transform=target_transform,
                )

    test_dataset = datasets.SVHN(
                    root=data_path, split='test', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                    #target_transform=target_transform
                )
    return train_dataset, test_dataset

def load_mnist_help():
    train_dataset = datasets.MNIST(root=data_path,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

    test_dataset = datasets.MNIST(root=data_path,
                              train=False,
                              transform=transforms.ToTensor())
    return train_dataset, test_dataset

def load_cifar10_help():
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    num_classes = 10


    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)

    return trainset, testset

def load_data(dataset, batch_size, data_path, n_train = None, n_test=None):
    """
    return the first n_train training data and first n_test testing data

    """
    batch_size = batch_size
    data_path = data_path
    if dataset == "MNIST":
        train_dataset, test_dataset = load_mnist_help()
    elif dataset == "SVHN":
        train_dataset, test_dataset = load_svhn_help()
    elif dataset == "CIFAR10":
        train_dataset, test_dataset = load_cifar10_help()
    else:
        raise fNotImplementedError

    # Data loader
    if n_train:
        train_dataset = Subset(train_dataset, range(n_train))

    if n_test:
        test_dataset = Subset(train_dataset, range(n_test))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    return train_loader, test_loader
