import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import metrics
from torch.autograd import Variable
import os, time
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import pandas as pd



path = Path('/home/ghazi/hsi-kegs/cloud-removal/uniform-light')
info = yaml.safe_load(open(path / 'info.yaml', 'r'))
bands = info['wavelength']
bands = np.array(bands).reshape(-1)

def to_psnr(cloud, gt):
    mse = F.mse_loss(cloud, gt, reduction='none')   
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    # intensity_max = 1
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(cloud, gt):
    cloud_list = torch.split(cloud, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    cloud_list_np = [cloud_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    ssim_list = [metrics.structural_similarity(cloud_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(cloud_list))]

    return ssim_list


def validation(net, val_data_loader, device, save_tag=False, noise_type = 'cloud'):
    """
    :param net: Network
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param save_tag: tag of saving image or not
    :return: average PSNR and SSIM value
    """
    psnr_list = []
    ssim_list = []
    cloudy_psnr_list = []
    cloudy_ssim_list = []
    cols = ['Original_PSNR', 'Predicted_PSNR', 'Original_SSIM', 'Predicted_SSIM']
    rows = []

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            cloud = val_data["Noise"]
            gt = val_data["Data"]
            cloud = cloud.to(device)
            gt = gt.to(device)
            cloud_removal ,_ ,_ = net(cloud)


            # --- Calculate the average PSNR --- #
            psnr = to_psnr(cloud_removal, gt)

            # --- Calculate the average SSIM --- #
            ssim = to_ssim_skimage(cloud_removal, gt)
            print(f"Image: {batch_id}")
            og_psnr = to_psnr(cloud, gt)
            og_ssim = to_ssim_skimage(cloud, gt)
            pred_psnr = to_psnr(cloud_removal, gt)
            pred_ssim = to_ssim_skimage(cloud_removal, gt)
            print(f"Original:::: PSNR: {og_psnr}, SSIM: {og_ssim}")
            print(f"Predicted Score:::: PSNR: {pred_psnr}, SSIM: {pred_ssim}")
            rows.append([og_psnr, pred_psnr, og_ssim, pred_ssim])
                   
        psnr_list.extend(psnr)
        ssim_list.extend(ssim)
        cloudy_psnr_list.extend(og_psnr)
        cloudy_ssim_list.extend(og_ssim)

        # --- Save image --- #
        if save_tag:
            path = './results/'
            if not os.path.exists(path):
                os.makedirs(path)
            save_image(cloud_removal, bands, path + str(batch_id))

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    cloudy_avr_psnr = sum(cloudy_psnr_list) / len(psnr_list)
    cloudy_avr_ssim = sum(cloudy_ssim_list) / len(ssim_list)
    rows.append([cloudy_avr_psnr, avr_psnr, cloudy_avr_ssim, avr_ssim])

    table = pd.DataFrame(data=rows, columns=cols)
    with open(f'outputs/{noise_type}_results.csv', 'w') as f:
        table.to_csv(f, index=False, float_format="%.5f")
    print(f"Results saved in results.csv")

    return avr_psnr, avr_ssim


def save_image(cloud_removal, image_name, path):
    cloud_removal = torch.split(cloud_removal, 1, dim=0)
    batch_num = len(cloud_removal)
    for ind in range(batch_num):
        print(cloud_removal[ind].shape)
        utils.save_image(cloud_removal[ind], path+'{}'.format(image_name[ind][:-3] + 'png'))



def print_log(epoch, num_epochs, train_psnr, train_ssim):
    print('Epoch [{0}/{1}], Train_PSNR:{2:.2f}, Train_SSIM:{3:.4f}'
          .format(epoch, num_epochs, train_psnr, train_ssim))

    # --- Write the training log --- #
    with open('./logs/train_log.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Train_PSNR: {3:.2f}, Train_SSIM: {4:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, train_ssim), file=f)




def rgb(image, hsi_bands):
    """
    Psuedo rgb image
    """
    
    red = np.where((hsi_bands >= 625) & (hsi_bands <= 700))[0]
    green = np.where((hsi_bands >= 500) & (hsi_bands <= 565))[0]
    blue = np.where((hsi_bands >= 450) & (hsi_bands <= 485))[0]

    red = image[..., red].mean(axis=-1)
    green = image[..., green].mean(axis=-1)
    blue = image[..., blue].mean(axis=-1)
    rgb = np.dstack([red, green, blue])
    return rgb
    
    
def rgb_image(image, bands):
    """
    Color-corrected rgb image
    """
    rgb_image = rgb(image, bands)
    # rgb_image = np.clip(rgb_image, 0, rgb_image.max())
    rgb_image = skimage.exposure.rescale_intensity(rgb_image, out_range=(0, 255)).astype(np.uint8)
    rgb_image = skimage.exposure.adjust_gamma(rgb_image, gamma=1, gain=1)
    return rgb_image.astype(np.uint8)




def save_image(image, bands, name):
    print(f"{name}: {image.shape}")
    image = torch.squeeze(image)
    image = image.detach().cpu().numpy()
    image = np.moveaxis(image, 0, 2)
    image = np.moveaxis(image, 0, 1)
    image = np.clip(image, a_min=0, a_max=None)
    image = rgb_image(image, bands)
    plt.imsave(name + '.png', image)

    plt.clf()