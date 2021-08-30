import torch
from models.base_model import BaseModel
from models import networks
import torch.nn as nn
import copy
from utils.util import tensor2im
import torch.nn as nn
import torchvision.transforms as transforms
import collections
import torch.nn.functional as F
from models import external_functions
import numpy as np
import os
import shutil
import cv2, random, imageio
        
class DIORBaseModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.n_human_parts = opt.n_human_parts
        self.n_style_blocks = opt.n_style_blocks
        # init_models
        self._init_models(opt)
        
        # loss
        if self.isTrain:
            self._init_loss(opt)
            
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--loss_coe_rec', type=float, default=2, help='n resnet transferring blocks in encoders')
        parser.add_argument('--loss_coe_per', type=float, default=0.2, help='n resnet transferring blocks in encoders')
        parser.add_argument('--loss_coe_sty', type=float, default=200, help='n resnet transferring blocks in encoders')
        parser.add_argument('--loss_coe_GAN', type=float, default=1, help='n resnet transferring blocks in encoders')
        parser.add_argument('--g2d_ratio', type=float, default=0.1, help='n resnet transferring blocks in encoders')
        parser.add_argument('--segm_dataset', type=str, default="")
        parser.add_argument('--netE', type=str, default='adgan', help='n resnet transferring blocks in encoders')
        parser.add_argument('--n_human_parts', type=int, default=8, help='n resnet transferring blocks in encoders')
        parser.add_argument('--n_kpts', type=int, default=18, help='n resnet transferring blocks in encoders')
        parser.add_argument('--n_style_blocks', type=int, default=4, help='n resnet transferring blocks in encoders')
        parser.add_argument('--style_nc', type=int, default=64, help='n resnet transferring blocks in encoders')
        return parser
    
        
    def _init_loss(self, opt):
        self.loss_names = ["G_GAN_pose", "G_GAN_content", 
                           "D_real_pose", "D_fake_pose", "D_real_content", "D_fake_content",
                           "rec", "per", "sty",]
        
        if self.isTrain:
            self.log_loss_update(reset=True)
            self.loss_coe = {'rec': opt.loss_coe_rec, 
                             'per':opt.loss_coe_per, 
                             'sty': opt.loss_coe_sty,
                             'GAN':opt.loss_coe_GAN
                             }
            
            # define loss functions
            self.criterionGAN = external_functions.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean').to(self.device)
            self.criterionMSE = torch.nn.MSELoss(reduction="mean").to(self.device)
            
  
    def _init_models(self, opt):
        self.model_names = ["E_attr", "G", "VGG"]
        self.frozen_models = ["VGG"]
        self.visual_names = ['from_img', 'fake_B', 'to_img']
        self.netVGG = networks.define_tool_networks(tool='vgg', load_ckpt_path="", gpu_ids=opt.gpu_ids)
        
        # netG
        self.netG = networks.define_G(input_nc=opt.n_kpts, output_nc=3, ngf=opt.ngf, latent_nc=opt.ngf * (2 ** 2), 
                                      style_nc=opt.style_nc,
                                      n_style_blocks=opt.n_style_blocks, n_human_parts=opt.n_human_parts, netG=opt.netG, 
                                      norm=opt.norm_type, relu_type=opt.relu_type,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
       
        self.netE_attr = networks.define_E(input_nc=3, output_nc=opt.style_nc, netE=opt.netE, ngf=opt.ngf, n_downsample=2,
                                           norm_type=opt.norm_type, relu_type=opt.relu_type, 
                                           init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        
        if self.isTrain:
            self.model_names += ["D_pose", "D_content"]
            self.netD_pose = networks.define_D(opt.n_kpts+3, opt.ndf, opt.netD,
                                          opt.n_layers_D, norm=opt.norm_type, use_dropout=not opt.no_dropout, 
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids
                                              )
            self.netD_content = networks.define_D(3+self.n_human_parts, opt.ndf, opt.netD,
                                          n_layers_D=3, norm=opt.norm_type, use_dropout=not opt.no_dropout, 
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        from_img, from_kpt, from_parse, to_img, to_kpt, to_parse, attr_label = input
        self.to_parse = to_parse.float().to(self.device)
        self.from_img = from_img.to(self.device)
        self.to_img = to_img.to(self.device)
        self.from_parse = from_parse.to(self.device)
        self.to_kpt = to_kpt.float().to(self.device) 
        self.from_kpt = from_kpt.float().to(self.device)
        self.attr_label = attr_label.long().to(self.device)
        self.to_parse2 = torch.cat([(self.to_parse == i).unsqueeze(1) for i in range(self.n_human_parts)], 1).float()

    def save_batch(self, save_dir, count, square=False):
        if False: #not square:
            self.from_img = F.interpolate(self.from_img, (256, 176))
            self.fake_B = self.to_img.clone() #F.interpolate(self.fake_B, (256, 176))
            self.to_img = F.interpolate(self.to_img, (256, 176))
        rets = torch.cat([self.from_img, self.to_img, self.fake_B], 3)
        for i, ret in enumerate(rets):
            count = count + 1
            img = tensor2im(ret)
            imageio.imwrite(os.path.join(save_dir, "generated_%d.jpg" % count), img)
        return count 

    def save_batch_single(self, save_dir, count):
        rets = self.fake_B
        for ret in rets:
            count = count + 1
            img = tensor2im(ret)
            imageio.imwrite(os.path.join(save_dir, "generated_%d.jpg" % count), img)
        return count 

    def compute_target_visuals(self):
        print_img = self.to_img.float().cpu().detach()
        print_img = (print_img + 1) / 2.0
        self.writer.add_images('target', print_img, 0)
          
    def encode_single_attr(self, img, parse, from_pose=None, to_pose=None, i=0):
        pass
    
    
    def encode_attr(self, img, parse, from_pose=None, to_pose=None):
        pass
    
    def decode(self, pose, attr_maps, attr_codes):
        pass
        

    def forward_viton(self, gid=5):
        pass

    
    def forward(self):
        pass

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D = self.compute_D_pose_loss() + self.compute_D_content_loss()
        
    def compute_D_pose_loss(self):
         # pose 
        fake_AB = torch.cat((self.to_kpt, self.fake_B), 1)  
        pred_fake = self.netD_pose(fake_AB.detach())
        self.loss_D_fake_pose = self.criterionGAN(pred_fake, False) * self.loss_coe['GAN']
        # Real
        real_AB = torch.cat((self.to_kpt, self.to_img), 1)
        pred_real = self.netD_pose(real_AB)
        self.loss_D_real_pose = self.criterionGAN(pred_real, True) * self.loss_coe['GAN']
        return (self.loss_D_fake_pose + self.loss_D_real_pose) / 0.5

    def compute_D_content_loss(self):
        # content
        fake_AB = torch.cat((self.to_parse2, self.fake_B), 1)  
        pred_fake = self.netD_content(fake_AB.detach())
        self.loss_D_fake_content = self.criterionGAN(pred_fake, False) * self.loss_coe['GAN']
        
        real_AB = torch.cat((self.to_parse2, self.to_img), 1)
        pred_real = self.netD_content(real_AB)
        self.loss_D_real_content = self.criterionGAN(pred_real, True)  * self.loss_coe['GAN']  
        return (self.loss_D_fake_content + self.loss_D_real_content) / 0.5
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # GAN loss
        fake_AB = torch.cat((self.to_kpt, self.fake_B), 1) 
        pred_fake = self.netD_pose(fake_AB)
        self.loss_G_GAN_pose = self.criterionGAN(pred_fake, True)

        fake_AB = torch.cat((self.to_parse2, self.fake_B), 1) 
        pred_fake = self.netD_content(fake_AB)
        self.loss_G_GAN_content = self.criterionGAN(pred_fake, True)

        self.loss_G = (self.loss_G_GAN_pose + self.loss_G_GAN_content) * self.loss_coe['GAN']

        
    def compute_rec_loss(self, pred, gt):
        self.loss_rec = 0.0
        if self.loss_coe['rec']:
            self.loss_rec = self.criterionL1(pred, gt) * self.loss_coe['rec']
        return self.loss_rec

    

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        
        self.set_requires_grad(self.netD_pose, True)  # enable backprop for D
        self.set_requires_grad(self.netD_content, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.loss_D.backward()
        self.optimizer_D.step()          # update D's weights
        
        # update G
        self.set_requires_grad(self.netD_pose, False) 
        self.set_requires_grad(self.netD_content, False) 
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.loss_G.backward()
        self.optimizer_G.step()             # udpate G's weights
        self.log_loss_update()