import torch.utils.data as data
import torch
from .human_parse_labels import get_label_map
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import copy, os, collections
import json
import random

def listdir(root):
    all_fns = []
    for fn in os.listdir(root):
        curr_root = os.path.join(root, fn)
        if os.path.isdir(curr_root):
            curr_all_fns = listdir(curr_root)
            all_fns += [os.path.join(fn, item) for item in curr_all_fns]
        else:
            all_fns += [fn]
    return all_fns
 
def create_texsyn_dataset(opt, isTrain=True):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
                transforms.Resize((296, 200)),
                transforms.RandomCrop((opt.crop_size)), 
                # transforms.RandomResizedCrop(opt.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    test_transform = transforms.Compose([
                transforms.Resize(opt.crop_size),
                transforms.ToTensor(),
                normalize,
            ])
    dataset = TextureDataset(dataroot=opt.dataroot, isTrain=isTrain, transform=train_transform if isTrain else test_transform)
    return dataset
    
    
class TextureDataset(data.Dataset):
    def __init__(self, dataroot, isTrain=True, transform=None):
        self.mask_dir = os.path.join(dataroot, "keypoints_heatmaps")
        self.isTrain = isTrain
        # if self.isTrain:
        self.tex_dir = os.path.join(dataroot, 'dtd/%s' % ("train" if isTrain else "test"))
        self.all_tex = [img for img in listdir(self.tex_dir) if img.endswith('.jpg') or  img.endswith('.png')]
      
        # mask
        self.aiyu2atr, self.atr2aiyu = get_label_map(n_human_part=4)
        
        # transforms
        self.transform = transform

    def _load_img(self, fn):
        img = Image.open(fn).convert("RGB")
        img = self.transform(img)
        return img
 
    def __len__(self):
        if self.isTrain:
            return 101966
        return len(self.all_tex)
    
    def __getitem__(self, index):
        
        tex_fn = self.all_tex[index % len(self.tex_dir)]
        tex = self._load_img(os.path.join(self.tex_dir, tex_fn))
        return tex
        
    
class TexSynDataset(data.Dataset):
    def __init__(self, dataroot, isTrain=True, crop_size=(256, 256)):
        self.mask_dir = os.path.join(dataroot, "keypoints_heatmaps")
        self.isTrain = isTrain
        if self.isTrain:
            self.all_masks = [ann + ".png" for ann in self._load_anns(dataroot)]
            self.tex_dir = os.path.join(dataroot, 'dtd/images')
            self.all_tex = [img for img in listdir(self.tex_dir) if img.endswith('.jpg') or  img.endswith('.png')]
        else:
            self.all_masks = [ann + ".png" for ann in self._load_anns(dataroot, False)]
            self.tex_dir = os.path.join(dataroot, "keypoints_heatmaps")
            self.all_tex = [ann + ".jpg" for ann in self._load_anns(dataroot, False)]
        
        # mask
        self.aiyu2atr, self.atr2aiyu = get_label_map(n_human_part=4)
        
        # transforms
        self.crop_size=crop_size
        self.resize = transforms.Resize(self.crop_size)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def _load_anns(self, dataroot, isTrain=True):
        if isTrain:
            tmp_fn = "gan_same1_train_pairs.txt"
            tmp_fn = "Anno/train_pairs.txt"#"Anno/train_pairs.txt"
            with open(os.path.join(dataroot, tmp_fn), "r") as f:
                anns = f.readlines()
            print("[dataset] load %d data from train split" % len(anns))
        else:
            tmp_fn = "Anno/test_pairs.txt"
            with open(os.path.join(dataroot, tmp_fn), "r") as f:
                anns = f.readlines()
            print("[dataset] load %d data from test split" % len(anns))
        anns = [ann.split(',')[0] for ann in anns]
        return anns
            
    def _load_img(self, fn):
        img = Image.open(fn).convert("RGB")
        img = self.resize(img)
        img = self.toTensor(img)
        img = self.normalize(img)
        return img
    
    def _load_mask(self, fn): 
        mask = Image.open(fn)
        mask = self.resize(mask)
        mask = torch.from_numpy(np.array(mask))
        
        texture_mask = copy.deepcopy(mask)
        for atr in self.atr2aiyu:
            aiyu = self.atr2aiyu[atr]
            texture_mask[texture_mask == atr] = aiyu
        return texture_mask
    
    
    def __len__(self):
        return len(self.all_masks)
    
    def __getitem__(self, index):
        tex_fn = self.all_tex[index % len(self.all_tex)]
        mask_fn = self.all_masks[index]
        tex = self._load_img(os.path.join(self.tex_dir, tex_fn))
        mask = self._load_mask(os.path.join(self.mask_dir, mask_fn))
        i = random.randint(0,3)
        mask = (mask == i).long().unsqueeze(0)
        return tex * mask, tex, mask
    
        
           
        
        