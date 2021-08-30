"""
Functions are modified on top of GFLA.
GFLA's license: https://github.com/RenYurui/Global-Flow-Local-Attention/blob/master/LICENSE.md

"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
import numpy as np

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss




def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

    

class MultiAffineRegularizationLoss(nn.Module):
    def __init__(self, kz_dic):
        super(MultiAffineRegularizationLoss, self).__init__()
        self.kz_dic=kz_dic
        self.method_dic={}
        for key in kz_dic:
            instance = AffineRegularizationLoss(kz_dic[key])
            self.method_dic[key] = instance
        self.layers = sorted(kz_dic, reverse=True) 
 
    def __call__(self, flow_fields):
        loss=0
        for i in range(len(flow_fields)):
            method = self.method_dic[self.layers[i]]
            loss += method(flow_fields[i])
        return loss

class AffineRegularizationLoss(nn.Module):
    """docstring for AffineRegularizationLoss"""
    # kernel_size: kz
    def __init__(self, kz):
        super(AffineRegularizationLoss, self).__init__()
        self.kz = kz
        self.criterion = torch.nn.L1Loss()
        from models.networks.block_extractor.block_extractor   import BlockExtractor
        from models.networks.local_attn_reshape.local_attn_reshape   import LocalAttnReshape
    
        self.extractor = BlockExtractor(kernel_size=kz)
        self.reshape = LocalAttnReshape()

        temp = np.arange(kz)
        A = np.ones([kz*kz, 3])
        A[:, 0] = temp.repeat(kz)
        A[:, 1] = temp.repeat(kz).reshape((kz,kz)).transpose().reshape(kz**2)
        AH = A.transpose()
        k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz**2) #K = (A((AH A)^-1)AH - I)
        self.kernel = np.dot(k.transpose(), k)
        self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz**2, kz, kz).unsqueeze(1)

    def __call__(self, flow_fields):
        grid = self.flow2grid(flow_fields)

        grid_x = grid[:,0,:,:].unsqueeze(1)
        grid_y = grid[:,1,:,:].unsqueeze(1)
        weights = self.kernel.type_as(flow_fields)
        #import pdb; pdb.set_trace()
        loss_x = self.calculate_loss(grid_x, weights)
        loss_y = self.calculate_loss(grid_y, weights)
        return loss_x+loss_y


    def calculate_loss(self, grid, weights):
        results = nn.functional.conv2d(grid, weights)   # KH K B [b, kz*kz, w, h]
        b, c, h, w = results.size()
        kernels_new = self.reshape(results, self.kz)
        f = torch.zeros(b, 2, h, w).type_as(kernels_new) + float(int(self.kz/2))
        grid_H = self.extractor(grid, f)
        result = torch.nn.functional.avg_pool2d(grid_H*kernels_new, self.kz, self.kz)
        loss = torch.mean(result)*self.kz**2
        return loss

    def flow2grid(self, flow_field):
        b,c,h,w = flow_field.size()
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(flow_field).float() 
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(flow_field).float()
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        return flow_field+grid

class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x, y, last_only=False, content_only=False):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        if not last_only:
            content_loss = 0.0
            content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
            content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
            content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
            content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
            content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
            if content_only:
                return content_loss

            # Compute loss
            style_loss = 0.0
            style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
            style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
            style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
            style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        else:
            content_loss = self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
            if content_only:
                return content_loss
            style_loss = self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return content_loss, style_loss

class PerceptualCorrectness(nn.Module):
    r"""

    """

    def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer  
        self.eps=1e-8 
        from models.networks.resample2d_package.resample2d import Resample2d
        self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=False):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)



        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        flow = F.interpolate(flow, [h,w])

        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        try:
            correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        except:
            print("An exception occurred")
            print(source_norm.shape)
            print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
        if mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
            mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3))
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)

        # print(correction_sample[0,2076:2082])
        # print(correction_max[0,2076:2082])
        # coor_x = [32,32]
        # coor = max_indices[0,32+32*64]
        # coor_y = [int(coor%64), int(coor/64)]
        # source = F.interpolate(self.source, [64,64])
        # target = F.interpolate(self.target, [64,64])
        # source_i = source[0]
        # target_i = target[0]

        # source_i = source_i.view(3, -1)
        # source_i[:,coor]=-1
        # source_i[0,coor]=1
        # source_i = source_i.view(3,64,64)
        # target_i[:,32,32]=-1
        # target_i[0,32,32]=1
        # lists = str(int(torch.rand(1)*100))
        # img_numpy = util.tensor2im(source_i.data)
        # util.save_image(img_numpy, 'source'+lists+'.png')
        # img_numpy = util.tensor2im(target_i.data)
        # util.save_image(img_numpy, 'target'+lists+'.png')
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
