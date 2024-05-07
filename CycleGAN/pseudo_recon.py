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
    
    # model = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    # model = torch.load(opt.weights)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

    pseudo_label_dic = {}
    prob_dic = {}

    for i, sample in enumerate(domain_loaderT):
        flag = False
        visualizer.reset() 
        target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
        target_pl = torch.stack([torch.from_numpy(refine_pseudo_label_dic.get(i)) for i in target_img_name])
        target_prob_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])
        target_label = target_label.to(device)
        target_pl = target_pl.to(device)
        target_prob_pl = target_prob_pl.to(device)
        _, target_new_pl, t = model.get_output_B(target_prob_pl, type1='one', type2='one')

        prob_dic[target_img_name] = target_new_pl.detach().cpu().numpy()
        pseudo_label_dic[target_img_name] = (target_new_pl>0.75).long().detach().cpu().numpy()

    if not os.path.exists('./log'):
        os.mkdir('./log')
    
    if args.datasetT=="Domain2":
        np.savez('./log/pseudolabel_D2_recon', pseudo_label_dic, prob_dic)

    if args.datasetT=="Domain4":
        np.savez('./log/pseudolabel_D4_recon', pseudo_label_dic, prob_dic)

    if args.datasetT=="Domain1":
        np.savez('./log/pseudolabel_D1_recon', pseudo_label_dic, prob_dic)   
