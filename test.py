import argparse
import torch
import torch.nn as nn  
from torch.utils.data import DataLoader, random_split
from model.network import VAE
from datasets.datasets import CloudRemovalDataset
from torchvision import transforms
from utils import to_psnr, validation
import yaml
from pathlib import Path
import numpy as np

torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--noise_type', type=str, default='dec_intensity', choices=['dec_intensity', 'cloud', 'gaussian', 'pernil'])
opt = parser.parse_args()


# ---  hyper-parameters for testing the neural network --- #
test_data_dir = './data/test/'
test_batch_size = 1
data_threads = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = Path('/home/ghazi/hsi-kegs/cloud-removal/uniform-light')
info = yaml.safe_load(open(path / 'info.yaml', 'r'))
bands = info['wavelength']
bands = np.array(bands).reshape(-1)

# --- Validation data loader --- #
dataset = CloudRemovalDataset(path, patch_size=256, noise_type=opt.noise_type)
train_ratio = 0.8
test_ratio = 0.2
total_size = len(dataset)
train_size = int(total_size*train_ratio)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader =  DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# --- Define the network --- #
model = VAE()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0])

# --- Load the network weight --- #
model.load_state_dict(torch.load('./checkpoints/cloud_removal.pth'))

# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
print('Length of test set:', len(test_dataloader))
test_psnr, test_ssim = validation(model, test_dataloader, device, save_tag=False, noise_type=opt.noise_type)
print('test_psnr: {0:.6f}, test_ssim: {1:.6f}'.format(test_psnr, test_ssim))
