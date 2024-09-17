import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Task: Implement a Convolutional Neural Network for image classification on the CIFAR10 dataset.
# 1. Define the CNN architecture
# 2. Implement the training loop
# 3. Evaluate the model on the test set

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO: Define the layers of the CNN

    def forward(self, x):
        # TODO: Implement the forward pass

# TODO: Set up data loaders for CIFAR10

# TODO: Define loss function and optimizer

# TODO: Implement training loop

# TODO: Evaluate the model on the test set

# Bonus: Implement data augmentation to improve performance