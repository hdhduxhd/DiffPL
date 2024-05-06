

import argparse
import os
import os.path as osp
import torch.nn.functional as F


import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import random
import numpy as np
import sys
sys.path[0]='/kaggle/working/DiffPL/cpr'

bceloss = torch.nn.BCELoss(reduction='none')
seed = 3377
savefig = False
get_hd = False
model_save = False#False
if True:
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval


from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='/kaggle/input/fundus-pretrain/sim_learn_D2.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--source', type=str, default='Domain1')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='/kaggle/input/fundus/Fundus')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--num_epochs',type=int,default=10)
    args = parser.parse_args()
    num_epochs = args.num_epochs
    tao=0.05
    gamma=2
    t_low = 0.4
    t_high = 0.85
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    
    train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    
    # 2. model
    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.train()

    if args.dataset=="Domain2":
        npfilename = '/kaggle/input/fundus-pseudo/pseudolabel_D2.npz'
        npfilename_new = '/kaggle/input/fundus-pseudo/pseudolabel_D2_new.npz'
    
    elif args.dataset=="Domain1":
        npfilename = '/kaggle/input/fundus-pseudo/pseudolabel_D1.npz'
        npfilename_new = '/kaggle/input/fundus-pseudo/pseudolabel_D1_new.npz'
    
    npdata = np.load(npfilename, allow_pickle=True)
    #uncertain_dic = npdata['arr_1'].item()
    proto_pseudo_dic = npdata['arr_2'].item()

    npdata = np.load(npfilename_new, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()
    prob_dic = npdata['arr_1'].item()


    optim_gen = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))#0.002
    #optim_gen.load_state_dict(checkpoint['optim_state_dict'])
    
    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0
    iter_num = 0
    for epoch_num in tqdm.tqdm(range(num_epochs), ncols=70):
        model.train()
        for batch_idx, (sample) in enumerate(train_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            prediction, _, feature, _, _ = model(data)
            prediction = torch.sigmoid(prediction)

            pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
            #uncertain_map = [uncertain_dic.get(key) for key in img_name]
            proto_pseudo = [proto_pseudo_dic.get(key) for key in img_name]
            prob = [prob_dic.get(key) for key in img_name]

            pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
            #uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float().cuda()
            proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float().cuda()
            prob = torch.from_numpy(np.asarray(prob)).float().cuda()

            for param in model.parameters():
                param.requires_grad = True
            optim_gen.zero_grad()
    
            mask = torch.zeros([pseudo_label.shape[0], 2, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask[prob<t_low] = 1.0#
            mask[prob>t_high] = 1.0#
            
            mask_proto = torch.zeros([data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()
            mask_proto[pseudo_label==proto_pseudo] = 1.0

            mask = mask*mask_proto
            
            loss_seg_pixel = bceloss(prediction, pseudo_label)
            #Focal loss, empirically found could make training stable, which is important 
            #since stopping epoch is hard to know without labelled data in target domain
            focal_coeff = ((1-prediction)**gamma)*pseudo_label + (prediction**gamma)*(1-pseudo_label)
            loss_seg_pixel = loss_seg_pixel * focal_coeff
            loss_seg = torch.sum(mask * loss_seg_pixel) / torch.sum(mask)
            #print(loss_seg.item())
            loss_seg.backward()
            optim_gen.step()
            iter_num = iter_num + 1
            
        #test
        model_eval = model
        model_eval.eval()
        val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
        cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
        with torch.no_grad():
            for batch_idx, (sample) in enumerate(test_loader):
                data, target, img_name = sample['image'], sample['map'], sample['img_name']
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                prediction, boundary, _, _, _ = model_eval(data)
                prediction = torch.sigmoid(prediction)

                target_numpy = target.data.cpu()
                prediction = prediction.data.cpu()
                prediction[prediction>0.75] = 1;prediction[prediction <= 0.75] = 0


                cup_dice = dice_coefficient_numpy(prediction[:,0, ...], target_numpy[:, 0, ...])
                disc_dice = dice_coefficient_numpy(prediction[:,1, ...], target_numpy[:, 1, ...])

                for i in range(prediction.shape[0]):
                    hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], get_hd)
                    if np.isnan(hd_tmp):
                        datanum_cnt_cup -= 1.0
                    else:
                        cup_hd += hd_tmp

                    hd_tmp = hd_numpy(prediction[i, 1, ...], target_numpy[i, 1, ...], get_hd)
                    if np.isnan(hd_tmp):
                        datanum_cnt_disc -= 1.0
                    else:
                        disc_hd += hd_tmp
                
                val_cup_dice += np.sum(cup_dice)
                val_disc_dice += np.sum(disc_dice)

                datanum_cnt += float(prediction.shape[0])
                datanum_cnt_cup += float(prediction.shape[0])
                datanum_cnt_disc += float(prediction.shape[0])

        val_cup_dice /= datanum_cnt
        val_disc_dice /= datanum_cnt
        cup_hd /= datanum_cnt_cup
        disc_hd /= datanum_cnt_disc
        

        if not os.path.exists('./logs'):
           os.mkdir('./logs')
        if args.dataset == 'Domain3':
            savefile = './log/cpr' + '_D3' + '.pth.tar'
        elif args.dataset == 'Domain2':
            savefile = './log/cpr' + '_D2' + '.pth.tar'
        elif args.dataset == 'Domain1':
            savefile = './log/cpr' + '_D1' + '.pth.tar'
        if (val_cup_dice+val_disc_dice)/2.0>best_avg:
            best_val_cup_dice = val_cup_dice; best_val_disc_dice = val_disc_dice; best_avg = (val_cup_dice+val_disc_dice)/2.0
            best_cup_hd = cup_hd; best_disc_hd = disc_hd; best_avg_hd = (best_cup_hd+best_disc_hd)/2.0
        

        print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
              (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
        print("best cup: %.4f best disc: %.4f best avg: %.4f best cup: %.4f best disc: %.4f best avg: %.4f" %
             (best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd))
        model.train()

    if model_save:
        torch.save({
            'model_state_dict': model_eval.state_dict(),
            'optim_state_dict':optim_gen.state_dict()
        }, savefile)

