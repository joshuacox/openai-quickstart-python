#!/usr/bin/env python
import torch
print("Torch CUDA is available: ", torch.cuda.is_available())
print("Torch CUDA device count = ", torch.cuda.device_count())
print("Torch CUDA current device = ", torch.cuda.current_device())
print("Torch CUDA current device name = ", torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X_train = torch.FloatTensor([0., 1., 2.])
print("X_train is", X_train.is_cuda)
X_train = X_train.to(device)
print("X_train is", X_train.is_cuda)


