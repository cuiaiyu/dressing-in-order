"""
This file is built upon CycleGAN, MUNIT code.
"""

import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import os

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + (opt.epoch_count - opt.n_epochs)/opt.lr_update_unit) / float(opt.n_epochs_decay / opt.lr_update_unit + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], do_init_weight=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        # import pdb; pdb.set_trace()
        assert(torch.cuda.is_available())
        
        net = net.cuda()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if do_init_weight:
        init_weights(net, init_type, init_gain=init_gain)
    return net

class ADGANEncoder(nn.Module):
    """"""
    def __init__(self, input_nc, output_nc, ngf=64, n_downsample=3, norm_type='none', relu_type='relu', frozen_flownet=True):
        
        super(ADGANEncoder, self).__init__()
        self.vgg_listen_list = ['conv_1_1', 'conv_2_1'] #, 'conv_3_1', 'conv_4_1']
        model = []
        model += [Conv2dBlock(input_nc, ngf, 7, 1, 3, norm_type, relu_type)]
        vgg_ngf = 64
        # n_downsample = 2
        for i in range(2):  # add downsampling layers
            mult = 2 ** i
            curr_ngf = ngf * mult
            curr_vgg = vgg_ngf * mult
            model += [Conv2dBlock(curr_ngf + curr_vgg, curr_ngf * 2, 3, 2, 1, norm_type, relu_type)]
        self.model = nn.Sequential(*model)
        
        latent_nc = ngf * 4
        self.segmentor = nn.Sequential(
            Conv2dBlock(latent_nc, latent_nc // 2, 3, 1, 1, 'instance', relu_type),
            Conv2dBlock(latent_nc // 2, latent_nc // 2, 3, 1, 1, 'instance', relu_type),
            Conv2dBlock(latent_nc // 2, 1, 3, 1, 1, 'none', 'none'),
            )
        if not frozen_flownet:
            from models.networks.block_extractor.block_extractor import BlockExtractor
            self.extractor = BlockExtractor(kernel_size=1)
        else:
            from utils.train_utils import torch_transform
            self.extractor = torch_transform

    
    def forward(self, x, vgg):
        """Standard forward"""
        v_layers = self.vgg_listen_list
        with torch.no_grad():
            vgg_out = vgg(x)
            
        # import pdb; pdb.set_trace()
        out = self.model[0](x)
        out = self.model[1](torch.cat([out, vgg_out[v_layers[0]]], dim=1)) # 128
        out = self.model[2](torch.cat([out, vgg_out[v_layers[1]]], dim=1)) # 256
        
        return out
    
    def enc_seg(self, x, flow, vgg):
        out = self(x, vgg)
        out = self.extractor(out, flow)
        
        attn = self.segmentor(out)
        attn = torch.sigmoid(attn)
        
        return out, attn
    
    def segm(self, x):
        attn = self.segmentor(x)
        attn = torch.sigmoid(attn)
        return attn
        
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res=0, input_dim=3, dim=64, norm='instance', activ='relu', pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        #if n_res > 0:
        #    self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # upsampling blocks
        #n_res = 0#6
        if n_res > 0:
            self.model += [ResBlocks(n_res, dim, norm='instance', activation=activ, pad_type=pad_type)]
        
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='layer', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        #return x
        return self.model(x)



##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='instance', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        if num_blocks == 0:
            self.model = Identity()
        else:
            for i in range(num_blocks):
                self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
            self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk=3, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
  
class StyleBlock(nn.Module):
    def __init__(self, out_nc=None, relu_type='relu'):
        super(StyleBlock, self).__init__()
        if relu_type == 'relu':
            self.relu = nn.ReLU(True) 
        elif relu_type == 'leakyrelu' :
            self.relu =  nn.LeakyReLU(0.2, True)
        else:
            self.relu = nn.ReLU6(True)
        self.ad_norm = AdaptiveInstanceNorm(out_nc)
        self.norm = nn.InstanceNorm2d(out_nc, affine=False, track_running_stats=False)
        latent_nc = out_nc
        
        self.conv1 = nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        
        
    def forward(self, x, style, mask=None, cut=False, adain=True):
        if len(style.size()) == 2:
            style = style[:,:, None, None]
        gamma, beta = style.chunk(2,1)
        gammas = gamma.chunk(2,1)
        betas = beta.chunk(2,1)
        
        out = x
        out = self.conv1(x)
        if adain:
            out = self.ad_norm(out, gammas[0], betas[0], mask)
        else:
            out = self.norm(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if adain:
            out = self.ad_norm(out, gammas[1], betas[1], mask)
        else:
            out = self.norm(out)
        if cut:
            return out
        return out + x      

        
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel, affine=False, track_running_stats=False)
        
    def forward(self, input, gamma, beta,mask=None):
        if not isinstance(mask, torch.Tensor): # mask == None:
            out = self.norm(input)
            out = gamma * out + beta
            return out   
        else:
            out = self.norm(input)
            out = gamma * out + beta
            return out * mask + input * (1 - mask)

        

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = (not norm == 'batch')
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = Identity()
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'none':
            self.activation = Identity()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'instance':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'layer':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = Identity()
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = Identity()
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x)
        x = self.activation(x)
        return x
    

class Identity(nn.Module):
    def forward(self, x):
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)