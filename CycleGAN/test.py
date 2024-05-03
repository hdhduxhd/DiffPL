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
import torch.nn.functional as F
import sys
sys.path[0]='/kaggle/working/DiffPL'
from CycleGAN.options.test_options import TestOptions
from CycleGAN.models import create_model
from CycleGAN.util.visualizer import Visualizer
from cpr.utils.metrics import *

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr


if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        #tr.Resize(512),###
        tr.RandomScaleCrop(256),
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
        tr.Resize(256),
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
    
    # model = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    # model = torch.load(opt.weights)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    visualizer = Visualizer(opt)

    dice_before_cup = 0
    dice_after_cup = 0
    dice_before_disc = 0
    dice_after_disc = 0

    nums = []
    names = []

    for i, sample in enumerate(domain_loaderT):
        flag = False
        visualizer.reset() 
        target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
        target_pl = torch.stack([torch.from_numpy(refine_pseudo_label_dic.get(i)) for i in target_img_name])
        target_prob_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])
        temp = {"image":target_image+1, "ground_truth":target_label, "pseudo_label":target_pl, "prob_pseudo_label":target_prob_pl}
        target_pl = F.interpolate(target_pl.float(), size=(256, 256), mode='bilinear', align_corners=False)
        target_prob_pl = F.interpolate(target_prob_pl, size=(256, 256), mode='bilinear', align_corners=False)
        target_label = target_label.to(device)
        target_pl = target_pl.to(device)
        target_pl[target_pl > 0.75] = 1
        target_pl[target_pl <= 0.75] = 0
        target_prob_pl = target_prob_pl.to(device)
        _, target_new_pl, t = model.get_output_B(target_prob_pl, type1='one', type2='one')
        temp["prob_new_pseudo_label"] = target_new_pl
        target_new_pl[target_new_pl > 0.75] = 1
        target_new_pl[target_new_pl <= 0.75] = 0
        visualizer.plot_current_metrics({"timestep":t[0][0]})
        
        dice_prob_cup, dice_prob_disc = dice_coeff_2label(target_pl, target_label)
        before_cup, before_disc = dice_prob_cup, dice_prob_disc
        dice_before_cup += dice_prob_cup
        dice_before_disc += dice_prob_disc
        visualizer.plot_current_metrics({"before_dice_cup":dice_prob_cup,"before_dice_disc":dice_prob_disc})

        dice_prob_cup, dice_prob_disc = dice_coeff_2label(target_new_pl, target_label)
        after_cup, after_disc = dice_prob_cup, dice_prob_disc
        dice_after_cup += dice_prob_cup
        dice_after_disc += dice_prob_disc
        visualizer.plot_current_metrics({"after_dice_cup":dice_prob_cup,"after_dice_disc":dice_prob_disc})

        if before_cup < after_cup:
            print('image:{%d}, before_cup{%.4f}<after_cup{%.4f}'%(i,before_cup,after_cup))
            flag = True
        if before_disc < after_disc:
            print('image:{%d}, before_disc{%.4f}<after_disc{%.4f}'%(i,before_disc,after_disc))
            flag = True
        if flag:
            nums.append(i)
            names.append(target_img_name)
            print(target_img_name)

        visualizer.display_current_results(temp, i, True)
        
    dice_before_cup /= len(domain_loaderT)
    dice_before_disc /= len(domain_loaderT)
    dice_after_cup /= len(domain_loaderT)
    dice_after_disc /= len(domain_loaderT)

    print('%.4f,%.4f'%(dice_before_cup,dice_after_cup))
    print('%.4f,%.4f'%(dice_before_disc,dice_after_disc))

    print("nums:",nums)
    print("names:",names)
