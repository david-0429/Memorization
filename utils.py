import numpy as np
from collections import defaultdict

import torch


# Gradient store
def grad_store(images, targets, model, loss_function):
    model.train()
    grad_dict = defaultdict(list)

    outputs = model(images)

    loss = loss_function(outputs, targets)
    loss.backward()

    # Extract gradients
    for i, (name, param) in enumerate(model.named_parameters()):
        if ('layer' in name) and ('conv' in name):
            key = name.split('.')[0]
            value = np.array(param.grad.clone().cpu())
            grad_dict[key].append(value)
    return grad_dict

#-------------------------------------------------------------------------------------------------------

# Calculate mean gradient of all batch
def calc_mean_grad(grad_batch_list):

  for i, batch_dict in enumerate(grad_batch_list):
    if i == 0:
        epoch_grad_dict = batch_dict.copy()

    else:
        for key, value in batch_dict.items():
            epoch_grad_dict[key] += value
  
  # Get mean grad vectors w.r.t. batch
    for key, value in epoch_grad_dict.items():
        epoch_grad_dict[key] = [x / len(grad_batch_list) for x in value]


  return epoch_grad_dict
