import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader, random_split
from model.network import VAE
from loss.edg_loss import edge_loss
from datasets.datasets import CloudRemovalDataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from utils import to_psnr, to_ssim_skimage, validation, print_log, save_image
import time
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import numpy as np

np.random.seed(1234)
torch.manual_seed(1234)


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
parser.add_argument('--samples', type=int, default=10, help='number of samples')
parser.add_argument('--noise_type', type=str, default='dec_intensity', choices=['dec_intensity', 'cloud', 'gaussian', 'pernil'])
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
print('===> Building model')
model = VAE()

# --- Define the MSE loss --- #
L1_Loss = nn.SmoothL1Loss(beta=0.5)
L1_Loss = L1_Loss.to(device)

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)

# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999)) 
scheduler = StepLR(optimizer,step_size= train_epoch // 2,gamma=0.1)

# --- Load training data and validation/test data --- #
path = Path('/home/ghazi/hsi-kegs/cloud-removal/uniform-light')
info = yaml.safe_load(open(path / 'info.yaml', 'r'))
bands = info['wavelength']
bands = np.array(bands).reshape(-1)


dataset = CloudRemovalDataset(path, patch_size=256, noise_type=opt.noise_type)
train_ratio = 0.8
test_ratio = 0.2
total_size = len(dataset)
train_size = int(total_size*train_ratio)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader =  DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

psnr = []
ssim = []
ok = False
# --- Training --- #
for epoch in range(1, opt.nEpochs + 1):
    print("Training...")
    start_time = time.time()
    psnr_list = []
    ssim_list = []
    for iteration, inputs in enumerate(train_dataloader, 1):

        cloud = inputs["Noise"]
        ref = inputs["Data"]
        cloud = cloud.to(device)
        ref = ref.to(device)
        # if epoch == 1:
        #     l1_loss = L1_Loss(cloud, ref)
        #     EDGE_loss = edge_loss(cloud, ref, device)
        #     psnr = to_psnr(cloud, ref)
        #     ssim = to_ssim_skimage(cloud, ref)
        #     print(f"l1_loss: {l1_loss}, Edge_Loss: {EDGE_loss}, PSNR: {psnr}, SSIM: {ssim}")

        # print(f"Cloud: {cloud.max()}")
        # print(f"Reference: {ref.max()}")    

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        model.train()
        cloud_removal, mean, log_var = model(cloud)

        # --- Multiple samples --- #
        for i in range(num_samples-1):
            c, m, l = model(cloud)
            cloud_removal = cloud_removal + c
            mean = mean + m
            log_var = log_var + l
        
        # --- Take the expectation --- #
        cloud_removal = cloud_removal / num_samples
        mean = mean / num_samples
        log_var = log_var / num_samples
        l1_loss = L1_Loss(cloud_removal, ref)
        kl_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        EDGE_loss = edge_loss(cloud_removal, ref, device)
        Loss = l1_loss + 0.01*kl_div + 0.18*EDGE_loss
        # Loss = kl_div
        Loss.backward()
        optimizer.step()
        # print(f"Difference bw pixels: {ref - cloud_removal}")
        if iteration % 5 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.5f} KL_div: {:.6f} L1_loss: {:.4f} EDGE_loss: {:.4f} Time: {:.2f}min".format(epoch, iteration, len(train_dataloader), Loss.item(), kl_div.item(), l1_loss.item(), EDGE_loss.item(), (time.time()-start_time)/60))
            
        # --- To calculate average PSNR and SSIM --- #
        # print(f"PSNR: {to_psnr(cloud_removal, ref)}, SSIM: {to_ssim_skimage(cloud_removal, ref)}")
        psnr_list.extend(to_psnr(cloud_removal, ref))
        ssim_list.extend(to_ssim_skimage(cloud_removal, ref))

        # if epoch == opt.nEpochs:
        #     print(f"Epoch: {epoch}")
        #     save_image(cloud_removal, bands, 'cloud_removal')
        #     save_image(cloud, bands, 'cloud')
        #     save_image(ref, bands, 'ref')


    scheduler.step()
    
    train_psnr = sum(psnr_list) / len(psnr_list)
    train_ssim = sum(ssim_list) / len(ssim_list)

    psnr.append(train_psnr)
    ssim.append(train_ssim)

    
    # --- Print log --- #
    print_log(epoch, train_epoch, train_psnr, train_ssim)

    if epoch%5 == 0:
        # --- Save the network  --- #
        print(f"Saving the model")
        torch.save(model.state_dict(), './checkpoints/cloud_removal.pth')


print(torch.any(cloud_removal < 0).item())
print(torch.any(cloud < 0).item())
print(torch.any(ref < 0).item())

print(cloud_removal)


# # Plot train_psnr 
# plt.plot(psnr)
# plt.xlabel('No-of-Epochs')
# plt.ylabel('PSNR')
# plt.savefig('psnr_10-3.png')

# plt.clf()   

# plt.plot(ssim)
# plt.xlabel('No-of-Epochs')
# plt.ylabel('SSIM')
# plt.savefig('ssim_10-3.png')