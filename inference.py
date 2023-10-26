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
import matplotlib.pyplot as plt




# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
parser.add_argument('--samples', type=int, default=10, help='number of samples')
opt = parser.parse_args()
print(opt)


# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = './data/train/'
train_batch_size = opt.batchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs
device_ids = GPUs_list
num_samples = opt.samples
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Loading model')
model = VAE()
model.eval()

# --- Define the MSE loss --- #
L1_Loss = nn.SmoothL1Loss(beta=0.5)
L1_Loss = L1_Loss.to(device)

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load('./checkpoints/cloud_removal_100.pth'))

# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999)) 
scheduler = StepLR(optimizer,step_size= train_epoch // 2,gamma=0.1)

# --- Load training data and validation/test data --- #
path = Path('/home/ghazi/hsi-kegs/cloud-removal')
train_dataset = CloudRemovalDataset(path, patch_size=32)
train_dataloader =  DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

psnr = []
ssim = []
ok = False






for iteration, inputs in enumerate(train_dataloader, 1):
    cloud = inputs["Noise"]
    ref = inputs["Data"]
    cloud = cloud.to(device)
    ref = ref.to(device)
    cloud_removal, mean, log_var = model(cloud)

    for i in range(num_samples-1):
        c, m, l = model(cloud)
        cloud_removal = cloud_removal + c
        mean = mean + m
        log_var = log_var + l
    cloud_removal = cloud_removal / num_samples
    cloud_removal = torch.squeeze(cloud_removal)
    ref = torch.squeeze(ref)
    print(ref.shape)
    plt.imshow(cloud_removal.permute(1, 2, 0).detach().cpu().numpy(),  cmap='pink')
    plt.savefig('cloud_removal.png')
    plt.clf()
    plt.imshow(ref.permute(1, 2, 0).detach().cpu().numpy(), cmap='pink')
    plt.savefig('ref.png')