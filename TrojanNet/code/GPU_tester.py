import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10 # CIFAR10 是一个由 60,000 张常见物体的 32x32 彩色图像组成的数据集。
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(torch.__version__)
print(torch.cuda.is_available())