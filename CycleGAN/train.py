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
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path[0]='/kaggle/working/DiffPL'
from CycleGAN.options.train_options import TrainOptions
# from data import create_dataset
from CycleGAN.models import create_model
from CycleGAN.util.visualizer import Visualizer
from cpr.utils.metrics import *

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)
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
    domain_iterS = iter(domain_loaderS)

    domain_T = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetT, split='train/ROIs', transform=composed_transforms_ts)
    domain_loaderT = DataLoader(domain_T, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    domain_iterT = iter(domain_loaderT)

    domain_val = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetT, split='test/ROIs', transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    npfilename = '/kaggle/input/fundus-pseudo/pseudolabel_D2.npz'
    npfilename_new = '/kaggle/input/fundus-pseudo/pseudolabel_D2_new.npz'
    refine_npdata = np.load(npfilename_new, allow_pickle=True)
    refine_pseudo_label_dic = refine_npdata['arr_0'].item()
    refine_prob_dic = refine_npdata['arr_1'].item()
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        for i in range(max(len(domain_loaderS),len(domain_loaderT))):
            try:
                sample = next(domain_iterS)
                source_image, source_label, source_img_name = sample['image'], sample['map'], sample['img_name']
            except Exception as err:
                domain_iterS = iter(domain_loaderS)
                sample = next(domain_iterS)
                source_image, source_label, source_img_name = sample['image'], sample['map'], sample['img_name']
            try:
                sample = next(domain_iterT)
                target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
                target_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])
            except Exception as err:
                domain_iterT = iter(domain_loaderT)
                sample = next(domain_iterT)
                target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
                target_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])

            # Resize target_pl to 1x2x256x256
            target_pl = F.interpolate(target_pl, size=(256, 256), mode='bilinear', align_corners=False)
            data = {"A": source_label.float(), "B": target_pl}
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            dice_cup, dice_disc = 0, 0
            max_dice = 0
            for sample in domain_loaderT:
                target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
                target_prob_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])
                target_label = target_label.to(device)
                target_prob_pl = target_prob_pl.to(device)
                target_new_pl = model.get_output_B(target_prob_pl, type1='one', type2='one')
                target_new_pl[target_new_pl > 0.5] = 1
                target_new_pl[target_new_pl <= 0.5] = 0
                dice_prob_cup, dice_prob_disc = dice_coeff_2label(target_new_pl, target_label)
                dice_cup += dice_prob_cup
                dice_disc += dice_prob_disc
            dice_cup /= len(domain_loaderT)    
            dice_disc /= len(domain_loaderT)
            if (dice_cup + dice_disc) / 2 > max_dice:
                max_dice = (dice_cup + dice_disc) / 2
                print('dice_cup: %.4f, dice_disc: %.4f' % (dice_cup, dice_disc))
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
