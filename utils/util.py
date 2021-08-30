"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import argparse

import torch.nn as nn

COLORS = np.array([
    [255,255,255], [66,135,245], [245,90,66], [245,206,66],
    [209,245,66], [105,245,66], [129,66,245], [245,66,203],
]) / 255.0 # green, pink, yellow

def downsampling(im, sx, sy):
    m = nn.AdaptiveAvgPool2d((round(sx),round(sy)))
    return m(im)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def assign_color(mask, n_labels):
    if len(mask.size()) == 4:
        mask = mask.sequeeze()
    N,H,W = mask.size()
    ret = []
    for i in range(n_labels):
        curr_parse = []
        for j in range(3):
            curr = (mask == i).float() * COLORS[i,j]
            curr_parse.append(curr.unsqueeze(1))
        ret += [torch.cat(curr_parse, 1)]
    return sum(ret)
       

def generate_zeros(min_h, max_h, min_w, max_w):
    h = random.randint(min_h, max_h - 1)
    w = random.randint(min_w, max_w - 1)
    zeros = torch.zeros((1,1,h,w))
    return zeros

def inject_zeros(img, margin=0.2, min_pad_size=0.2, max_pad_size=0.4):
    N,C,H,W = img.size()
    # generate pad
    min_h, min_w = max(1, int(min_pad_size * H)), max(1, int(min_pad_size * W))
    max_h, max_w = int(max_pad_size * H), int(max_pad_size * W)
    zeros = generate_zeros(min_h, max_h, min_w, max_w).to(img.device)
    
    # insert pad
    _,_,h,w = zeros.size()
    min_left, max_left = int(margin * W), int(W - margin * W - w)
    min_top, max_top = int(margin * H), int(H - margin * H - h)
    
    left = random.randint(min_left, max_left - 1)
    top = random.randint(min_top, max_top - 1)
    zeros = zeros.expand(N,C,h,w)
    img[:,:,top:top+h, left:left+w] = zeros
    return img

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             # print(kv)
             k,v = kv.split("=")
             my_dict[k] = int(v)
         setattr(namespace, self.dest, my_dict)   

class StoreList(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
        my_list = [int(item) for item in values.split(',')]
        setattr(namespace, self.dest, my_list)   
#           

def tensor2im(input_image, imtype=np.uint8, max_n=4):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if len(image_tensor.size()) == 4:
            all_image = [image_tensor[i] for i in range(min(image_tensor.size(0),max_n))]
            all_image = torch.cat(all_image, 1)
        else:
            all_image = image_tensor
        image_numpy = all_image.cpu().float().numpy() 
        #image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)



def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
