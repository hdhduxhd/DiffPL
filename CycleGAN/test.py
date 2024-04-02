"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path[0]='/kaggle/working/DiffPL'
from CycleGAN.options.test_options import TestOptions
from CycleGAN.models import networks
from cpr.utils.metrics import *

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr


if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    opt.add_argument("--weights", type=str, default='/kaggle/input/fundus_pl_refine/pytorch/resnet_9blocks/1/200_net_G_B.pth')
    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        #tr.Resize(512),###
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        # tr.RandomCrop(512),
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetS, split='train/ROIs', transform=composed_transforms_ts)
    domain_loaderS = DataLoader(domain, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    domain_T = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetT, split='train/ROIs', transform=composed_transforms_ts)
    domain_loaderT = DataLoader(domain_T, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    domain_val = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetS, split='test/ROIs', transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    npfilename = '/kaggle/input/fundus-pseudo/pseudolabel_D2.npz'
    npfilename_new = '/kaggle/input/fundus-pseudo/pseudolabel_D2_new.npz'
    refine_npdata = np.load(npfilename_new, allow_pickle=True)
    refine_pseudo_label_dic = refine_npdata['arr_0'].item()
    refine_prob_dic = refine_npdata['arr_1'].item()
    
    model = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    model.load_state_dict(torch.load(opt.weights))
    model.eval()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

    dice_before_cup = 0
    dice_after_cup = 0
    dice_before_disc = 0
    dice_after_disc = 0

    for sample in domain_loaderT:
        target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
        target_pl = torch.stack([torch.from_numpy(refine_pseudo_label_dic.get(i)) for i in target_img_name])
        target_prob_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])
        target_label = target_label.to(device)
        target_pl = target_pl.float().to(device)
        target_prob_pl = target_prob_pl.to(device)
        target_new_pl = model(target_prob_pl)
      
        dice_prob_cup, dice_prob_disc = dice_coeff_2label(target_pl, target_label)
        dice_before_cup += dice_prob_cup
        dice_before_disc += dice_prob_disc

        dice_prob_cup, dice_prob_disc = dice_coeff_2label(target_new_pl, target_label)
        dice_after_cup += dice_prob_cup
        dice_after_disc += dice_prob_disc

    dice_before_cup /= len(domain_loaderT)
    dice_before_disc /= len(domain_loaderT)
    dice_after_cup /= len(domain_loaderT)    
    dice_after_disc /= len(domain_loaderT)

    print('%.4f,%.4f'%(dice_before_cup,dice_after_cup))
    print('%.4f,%.4f'%(dice_before_disc,dice_after_disc))