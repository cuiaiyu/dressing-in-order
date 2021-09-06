import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import copy, os, collections
import json
from .human_parse_labels import get_label_map, DF_LABEL, YF_LABEL
import pandas as pd
from utils import pose_utils

TEST_PATCHES = [
    'chequered/chequered_0052.jpg','dotted/dotted_0072.jpg',
    'paisley/paisley_0015.jpg','striped/striped_0011.jpg'
]

class DFPairDataset(data.Dataset):

    def get_paths(self, root, phase, viton=False):
        pairLst = os.path.join(root, 'fasion-pairs-%s.csv' % phase)
        if viton:
            pairLst = os.path.join(root, 'fasion-pairs-%s.csv' % 'viton')
        name_pairs = self.init_categories(pairLst)
        
        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' % phase)

        return image_dir, bonesLst, name_pairs

    def init_categories(self, pairLst):
        with open(pairLst) as f:
            anns = f.readlines()
        anns = [line[:-1].split(",")[1:] for line in anns[1:]]
        return anns 


    def __init__(self, dataroot, dim=(256,256), isTrain=True, n_human_part=8, viton=False):
        super(DFPairDataset, self).__init__()
        self.root = dataroot
        self.isTrain = isTrain
        self.split = 'train' if isTrain else 'test'
        self.n_human_part = n_human_part
        self.dim = dim
        self._init(viton)
        self.mask_dir = self.root + "/%sM_lip" % ('train' if isTrain else 'test')
        
    def _init(self, viton):
        self.image_dir, self.bone_file, self.name_pairs = self.get_paths(self.root, self.split, viton)
        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')
        

        self.aiyu2atr, self.atr2aiyu = get_label_map(self.n_human_part)

        self.load_size = self.dim
        self.crop_size = self.load_size
    
        # transforms
        self.resize = transforms.Resize(self.crop_size)
        self.toTensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def __len__(self):
        return len(self.name_pairs)
    
    def _load_img(self, fn):
        img = Image.open(fn).convert("RGB")
        img = self.resize(img)
        img = self.toTensor(img)
        img = self.normalize(img)
        
        return img
    
    def _load_mask(self, fn): 
        mask = Image.open(fn + ".png")
        mask = self.resize(mask)
        mask = torch.from_numpy(np.array(mask))
        
        texture_mask = copy.deepcopy(mask)
        for atr in self.atr2aiyu:
            aiyu = self.atr2aiyu[atr]
            texture_mask[mask == atr] = aiyu
        return texture_mask
    
    def _load_kpt(self, name):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose  = pose_utils.cords_to_map(array, self.load_size, (256, 176))
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose  
        
    
    def get_to_item(self, key):
        img = self._load_img(os.path.join(self.image_dir,key))
        kpt = self._load_kpt(key)     
        parse = self._load_mask(os.path.join(self.mask_dir, key[:-4]))
        return img, kpt, parse
    
    def __getitem__(self, index):
        from_key, to_key = self.name_pairs[index]
        from_img, from_kpt, from_parse = self.get_to_item(from_key)
        to_img, to_kpt, to_parse = self.get_to_item(to_key)
            
        return from_img, from_kpt, from_parse, to_img, to_kpt, to_parse, index #torch.Tensor([0])
        
    
        
class DFVisualDataset(DFPairDataset):

    def __init__(self, dataroot, dim=(256,256), texture_dir="",isTrain=False, n_human_part=8):
        DFPairDataset.__init__(self, dataroot, dim, isTrain, n_human_part=n_human_part)
        # load anns
        # import pdb; pdb.set_trace()
        eval_anns_path = os.path.join(dataroot,"standard_test_anns.txt")
        self._load_visual_anns(eval_anns_path)
        # load standard pose
        self._load_standard_pose()
        #load standard patches
        #patch_root = "/".join(dataroot.split("/")[:-1])
        #self.standard_patches = [self._load_img(os.path.join(patch_root, "dtd/images", fn)).unsqueeze(0) for fn in TEST_PATCHES]
        #self.standard_patches = torch.cat(self.standard_patches)
        self.selected_keys = [ "gfla", "jacket", "lace", "pattern", "plaid", "plain", "print", "strip", "flower"]
        self.image_dir = dataroot +  "/test"
        self.mask_dir = dataroot +  "/testM_lip"

    def _load_standard_pose(self):
        self.standard_poses = [
           self._load_kpt(key).unsqueeze(0) for key in self.pose_keys
          ]
        self.standard_poses = torch.cat(self.standard_poses)

    def get_patches(self):
        return self.standard_patches
    def __len__(self):
        return sum([len(self.attr_keys[i]) for i in self.attr_keys])

    
    def _load_visual_anns(self, eval_anns_path):
        with open(eval_anns_path) as f:
            raw_anns = f.readlines()
        pose_cnt = 1
        self.pose_keys = []
        for line in raw_anns[1:]:
            if line.startswith("attr"):
                break
            self.pose_keys.append(line[:-1])
            pose_cnt += 1
        self.attr_keys = collections.defaultdict(list)
        #import pdb; pdb.set_trace()
        for line in raw_anns[pose_cnt+1:]:
            category, key = line[:-1].split(", ")
            self.attr_keys[category].append(key)
        mixed = []
        for category in ['flower','plaid','print','strip']:
            mixed.append(self.attr_keys[category][0])
        self.attr_keys["mixed"] = mixed

    def get_patch_input(self):
        return torch.cat(self.standard_patches)

    def get_all_pose(self, key, std_pose=True):
        if std_pose:
            return self.standard_poses
        folder_path = os.path.join(self.kpt_dir,key).split("/")
        prefix = folder_path[-1]
        folder_path = "/".join(folder_path[:-1])
        ret = []
        
        for fn in os.listdir(folder_path):
            if fn.startswith(prefix) and fn.endswith('_kpt.npy'):
                curr = self._load_kpt(os.path.join(folder_path, fn[:-8]))   
                ret.append(curr[None])
            
        if len(ret) < 2:
            return self.standard_poses
        return torch.cat(ret)
        
        
    def get_pose_visual_input(self, subset="plain", std_pose=True, view_postfix="_1_front"):
        keys = self.attr_keys[subset]
        keys = keys[:min(len(keys), 8)]
        all_froms, all_kpts, all_parses = [], [], []
        all_from_kpts = []
        for key in keys:
            curr_key = key# + view_postfix
            curr_from, curr_from_kpt, curr_parse = self.get_to_item(curr_key)
            all_from_kpts += [curr_from_kpt]
            curr_kpt = self.get_all_pose(key, std_pose=std_pose)
            all_kpts += [curr_kpt]
            all_froms += [curr_from.unsqueeze(0)]
            all_parses += [curr_parse.unsqueeze(0)]
        all_froms = torch.cat(all_froms)
        all_parses = torch.cat(all_parses)
        all_from_kpts = torch.cat(all_from_kpts)
        return all_froms, all_parses, all_from_kpts, all_kpts #self.standard_poses

    def get_attr_visual_input(self, subset="plain",view_postfix="_1_front"):
        keys = self.attr_keys[subset]
        keys = keys[:min(len(keys), 4)]
        all_froms, all_parses, all_kpts = [], [], []
        for key in keys:
            curr_key = key# + view_postfix
            curr_from, to_kpt, curr_parse = self.get_to_item(curr_key)
            all_froms += [curr_from.unsqueeze(0)]
            all_parses += [curr_parse.unsqueeze(0)]
            all_kpts += [to_kpt.unsqueeze(0)]
        all_froms = torch.cat(all_froms)
        all_parses = torch.cat(all_parses)
        all_kpts = torch.cat(all_kpts)
        return all_froms, all_parses, all_kpts 

    def get_inputs_by_key(self, key):
        #keys = self.attr_keys[subset]
        #keys = keys[:min(len(keys), 4)]
        curr_from, to_kpt, curr_parse = self.get_to_item(key)
        return curr_from, curr_parse, to_kpt
    
       