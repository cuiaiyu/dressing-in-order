import torch
import torch.nn as nn
from torchvision.models import vgg19, vgg16,resnet50
from collections import OrderedDict

# VGG 19
vgg_layer = {
     'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16, 'pool_3': 18, 'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25, 'pool_4': 27, 'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34, 'pool_5': 36
 }

vgg_layer_inv = {
     0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'conv_3_4', 18: 'pool_3', 19: 'conv_4_1', 21: 'conv_4_2', 23: 'conv_4_3', 25: 'conv_4_4', 27: 'pool_4', 28: 'conv_5_1', 30: 'conv_5_2', 32: 'conv_5_3', 34: 'conv_5_4', 36: 'pool_5'
 }

def count_parameters(model):
    return '%.2f M'% (sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)

class VGG_Model(nn.Module):
    def __init__(self, load_ckpt_path="", listen_list=[]):
        super(VGG_Model, self).__init__()
        vgg = vgg19(pretrained=True)
        if load_ckpt_path:
            print("load vgg ckpt from %s" % load_ckpt_path)
            weights = torch.load(load_ckpt_path)
            if "vgg_conv" in load_ckpt_path: # yifang's vgg
                # rename weights
                ckpt = OrderedDict()
                for key in weights:
                    new_key = vgg_layer['conv_'+ key[4:7]]
                    ckpt[str(new_key) + key[7:]] = weights[key].float()
                # load weights
                self.vgg_model = vgg.features
                self.vgg_model.load_state_dict(ckpt)
            elif 'vgg19-dcbb9e9d.pth' in load_ckpt_path:
                vgg.load_state_dict(torch.load(load_ckpt_path))
                # listen_list = ['conv_1_1', 'conv_2_1', 'conv_3_1','conv_4_1',]
                self.vgg_model = vgg.features
            else:
                for key in weights:
                    weights[key] = weights[key].float()
                vgg.load_state_dict(weights, strict=False)
                self.vgg_model = vgg.features
        else:
            print("load vgg ckpt from torchvision dict.")
            self.vgg_model = vgg.features
        last_needed_layer = vgg_layer[listen_list[-1]]
        needed_model = []
        cnt = 0
        for layer in self.vgg_model:
            needed_model.append(layer)
            cnt += 1
            if cnt > last_needed_layer + 1:
                break
        self.vgg_model = nn.Sequential(*needed_model)
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        self.listen = set()
        for layer in listen_list:
            self.listen.add(vgg_layer[layer])
        
        # normalization factors
        self.mean = torch.autograd.Variable(torch.FloatTensor([0.485, 0.456, 0.406])).view(1, 3, 1, 1)
        self.std = torch.autograd.Variable(torch.FloatTensor([0.229, 0.224, 0.225])).view(1, 3, 1, 1)

    def norm_im(self, im):
        device = im.device
        im = (im + 1) / 2
        im = (im - self.mean.to(device)) / self.std.to(device)
        return im

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.norm_im(x)
        features = OrderedDict()
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index - 1 in self.listen:
                features[vgg_layer_inv[index - 1]] = x
        return features
