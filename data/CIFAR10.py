import torch
import torchvision

from data.data_utils import transformed


#CIFAR10 mean, std
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD =  (0.247, 0.243, 0.261)

data_path = '/content/data/CIFAR10'


def CIFAR10_loader(args, is_train=True):
  global transform
  transform = transformed(args, CIFAR10_TRAIN_MEAN , CIFAR10_TRAIN_STD, train=is_train)
  
  if is_train:
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
  
  else:
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  
  return data_loader
