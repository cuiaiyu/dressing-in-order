from datasets.deepfashion_datasets import * 
from datasets.texture_synthesis_datasets import *
import torch.utils.data as data
import os

def create_dataset(opt, viton=False):
    print(opt.crop_size)
    Dataset = DFPairDataset
    data_dir = os.path.join(opt.dataroot)
    dataset = Dataset(dataroot=data_dir, dim=opt.crop_size, isTrain=(opt.phase == 'train'), n_human_part=opt.n_human_parts, viton=viton)
    loader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=(opt.phase == 'train'), num_workers=opt.n_cpus, pin_memory=True)
    return loader

def create_tex_dataloader(opt):
    isTrain = opt.phase == 'train'
    ds = create_texsyn_dataset(opt, isTrain)
    loader = data.DataLoader(ds, batch_size=opt.batch_size,  shuffle=isTrain,  num_workers=opt.n_cpus, pin_memory=True)
    return loader

def create_visual_ds(opt):
    Dataset = DFVisualDataset
    data_dir = os.path.join(opt.dataroot)
    return Dataset(dataroot=data_dir, dim=opt.crop_size, n_human_part=opt.n_human_parts)