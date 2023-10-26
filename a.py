import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import VAE
from loss.edg_loss import edge_loss
from datasets.datasets import CloudRemovalDataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from utils import to_psnr, to_ssim_skimage, validation, print_log
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

print(np.random.normal(0, 0.1))
def f(a):
    a = [1, 2, 4, 3]
    return a


x = [1, 2, 3, 4]
y = f(x)
print(x, y)