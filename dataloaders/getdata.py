import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr

# 1. dataset
composed_transforms_tr = transforms.Compose([
    tr.Resize(512),###
    # tr.RandomScaleCrop(512),
    tr.RandomRotate(),
    tr.RandomFlip(),
    # tr.elastic_transform(),
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

def get_loader(data_dir, datasetS, datasetT, batch_size):
    domain = DL.FundusSegmentation(base_dir=data_dir, dataset=datasetS, split='train/ROIs', transform=composed_transforms_tr)
    domain_loaderS = DataLoader(domain, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    domain_T = DL.FundusSegmentation(base_dir=data_dir, dataset=datasetT, split='train/ROIs', transform=composed_transforms_tr)
    domain_loaderT = DataLoader(domain_T, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    domain_val = DL.FundusSegmentation(base_dir=data_dir, dataset=datasetS, split='test/ROIs', transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return domain_loaderS, domain_loaderT, domain_loader_val

def get_transforms():
    return composed_transforms_tr, composed_transforms_ts
