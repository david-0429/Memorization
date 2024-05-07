import torch
import torchvision

from data.data_utils import transformed


#CIFAR100 mean, std
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

data_path = '/content/data/CIFAR100'


def CIFAR100_loader(args, is_train=True):
  global transform
  transform = transformed(args, CIFAR100_TRAIN_MEAN , CIFAR100_TRAIN_STD, train=is_train)
  
  if is_train:
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
  
  else:
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  
  return data_loader
