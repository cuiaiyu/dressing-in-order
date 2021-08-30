import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import random
from models.networks.base_networks import *
import os

class BaseGenerator(nn.Module):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64, n_human_parts=8, n_downsampling=2, n_style_blocks=4, norm_type='instance', relu_type='relu'):
        super(BaseGenerator, self).__init__()
        self.n_style_blocks = n_style_blocks
        self.n_human_parts = n_human_parts

        self.to_emb = ContentEncoder(n_downsample=n_downsampling, input_dim=18, dim=ngf,norm='instance', activ=relu_type, pad_type='zero')
        self.to_rgb = Decoder(n_upsample=n_downsampling, n_res=6, dim=latent_nc, output_dim=3)
        self.style_blocks = nn.Sequential(*[StyleBlock(latent_nc, relu_type=relu_type) for i in range(n_style_blocks)])

        self.fusion = nn.Sequential(
            Conv2dBlock(latent_nc + 1, latent_nc, 3, 1, 1, norm_type, relu_type),
            Conv2dBlock(latent_nc, latent_nc * 4, 3, 1, 1, 'none', 'none'),
            )

class DIORGenerator(BaseGenerator):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64, n_human_parts=4, n_downsampling=2, n_style_blocks=2, norm_type='instance', relu_type='relu', **kwargs):
        super(DIORGenerator, self).__init__(img_nc, kpt_nc, ngf, latent_nc, style_nc, 
        n_human_parts, n_downsampling, n_style_blocks, norm_type, relu_type, **kwargs)
      
        

    def attend_person(self, psegs, gmask):
        styles = [a for a,b in psegs]
        mask = [b for a,b in psegs]
        
        N,C,_,_ = styles[0].size()
        style_skin = sum(styles).view(N,C,-1).sum(-1)
        
        N,C,_,_ = mask[0].size()
        human_mask = sum(mask).float().detach()
        area = human_mask.view(N,C,-1).sum(-1) + 1e-5
        style_skin = (style_skin / area)[:,:,None,None]
        
        full_human_mask = sum([m.float() for m in mask[1:] + gmask]).detach() 
        full_human_mask = (full_human_mask > 0).float()
        style_human =  style_skin * full_human_mask # + styles[0]
        style_human = self.fusion(torch.cat([style_human, full_human_mask], 1))
        style_bg = self.fusion(torch.cat([styles[0], mask[0]], 1))        
        style = style_human * full_human_mask + (1 - full_human_mask) * style_bg 
        
        return style


    def attend_garment(self, gsegs, alpha=0.5):

        ret = []
        styles = [a for a,b in gsegs]
        attns = [b for a,b in gsegs]
        
        for s,attn in zip(styles, attns):
            attn = (attn > alpha).float().detach()
            s = F.interpolate(s, (attn.size(2), attn.size(3)))
            N,C,H,W = s.size()
            mean_s = s.view(N,C,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
            s = s + mean_s
            s = self.fusion(torch.cat([s, attn], 1))
            s = s * attn
            ret.append(s)
            
        return ret, attns

    def forward(self, pose, psegs, gsegs, alpha=0.5):

        style_fabrics, g_attns = self.attend_garment(gsegs, alpha=alpha)
        style_human  = self.attend_person(psegs, g_attns)
        
        pose = self.to_emb(pose)
        out = pose
        for k in range(self.n_style_blocks // 2):
            out = self.style_blocks[k](out, style_human)

        base = out
        self.base = base

        for i in range(len(g_attns)): 
            attn = g_attns[i]
            curr_mask = (attn > alpha).float().detach()
            N = curr_mask.size(0)
            exists = torch.sum(curr_mask.view(N,-1), 1)
            exists = (exists > 0)[:,None,None,None].float()
            
            attn = exists * curr_mask * attn # * fattn
            
            for k in range(self.n_style_blocks // 2, self.n_style_blocks):
                base0 = out
                out = self.style_blocks[k](out, style_fabrics[i],cut=True) 
                out = out * attn + base0 * (1 - attn)
                
            
        fake = self.to_rgb(out)
        return fake
    
class DIORv1Generator(BaseGenerator):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64, n_human_parts=4, n_downsampling=2, n_style_blocks=2, norm_type='instance', relu_type='relu', **kwargs):
        super(DIORv1Generator, self).__init__(img_nc, kpt_nc, ngf, latent_nc, style_nc, 
        n_human_parts, n_downsampling, n_style_blocks, norm_type, relu_type, **kwargs)
      
        

    def attend_person(self, psegs, gmask):
        styles = [a for a,b in psegs]
        mask = [b for a,b in psegs]
        
        N,C,_,_ = styles[0].size()
        style_skin = sum(styles).view(N,C,-1).sum(-1)
        
        N,C,_,_ = mask[0].size()
        human_mask = sum(mask).float().detach()
        area = human_mask.view(N,C,-1).sum(-1) + 1e-5
        style_skin = (style_skin / area)[:,:,None,None]
        
        full_human_mask = sum([m.float() for m in mask[1:] + gmask]).detach() 
        full_human_mask = (full_human_mask > 0).float()
        style_human =  style_skin * full_human_mask + styles[0] # typo here
        style_human = self.fusion(torch.cat([style_human, full_human_mask], 1))
        style_bg = self.fusion(torch.cat([styles[0], mask[0]], 1))        
        style = style_human * full_human_mask + (1 - full_human_mask) * style_bg 
        
        return style


    def attend_garment(self, gsegs, alpha=0.5):

        ret = []
        styles = [a for a,b in gsegs]
        attns = [b for a,b in gsegs]
        
        for s,attn in zip(styles, attns):
            attn = (attn > alpha).float().detach()
            s = F.interpolate(s, (attn.size(2), attn.size(3)))
            N,C,H,W = s.size()
            mean_s = s.view(N,C,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
            s = s + mean_s
            s = self.fusion(torch.cat([s, attn], 1))
            s = s * attn
            ret.append(s)
            
        return ret, attns

    def forward(self, pose, psegs, gsegs, alpha=0.5):

        style_fabrics, g_attns = self.attend_garment(gsegs, alpha=alpha)
        style_human  = self.attend_person(psegs, g_attns)
        
        pose = self.to_emb(pose)
        out = pose
        for k in range(self.n_style_blocks // 2):
            out = self.style_blocks[k](out, style_human)

        base = out
        self.base = base

        for i in range(len(g_attns)): 
            attn = g_attns[i]
            curr_mask = (attn > alpha).float().detach()
            N = curr_mask.size(0)
            exists = torch.sum(curr_mask.view(N,-1), 1)
            exists = (exists > 0)[:,None,None,None].float()
            
            attn = exists * curr_mask * attn # * fattn
            
            for k in range(self.n_style_blocks // 2, self.n_style_blocks):
                base0 = out
                out = self.style_blocks[k](out, style_fabrics[i],cut=True) 
                out = out * attn + base0 * (1 - attn)
                
            
        fake = self.to_rgb(out)
        return fake
    


