import numpy as np
import random

import torch
import torchvision.transforms as transforms


def transformed(args, mean, std, train=True):

    if train:
      if args.DA == "none":
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

      elif args.DA == "flip_crop":
          transform = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    
    return transform

#-------------------------------------------------------------------------------------------------------

def make_noisy_label(true_labels, cls_num):

    noisy_label = []
    for t_l in true_labels:
        label_list = np.arange(cls_num)

        # Delete the true label within whole label list
        label_list = np.delete(label_list, int(t_l))
        noisy_label.append(random.choice(label_list))

    noisy_labels = torch.tensor(noisy_label)
    return noisy_labels.cuda()
