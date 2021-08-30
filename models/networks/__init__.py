from .base_networks import *
from .generators import *
from .vgg import *
from .gfla import PoseFlowNet, ResDiscriminator
import importlib

def find_generator_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models.networks.generators"
    #import pdb; pdb.set_trace()
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'Generator'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower(): \
           #and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def define_tool_networks(tool, load_ckpt_path="", gpu_ids=[], init_type='kaiming', init_gain=0.02):
    if tool == 'shape_cls':
        net = OutfitShapeClassifer(img_opt='vgg19', n_tasks=len(ALL_CATA), n_labels_per_task=[ALL_CATA[i] for i in ALL_CATA])
        if load_ckpt_path:
            ckpt = torch.load(load_ckpt_path)
            net.load_state_dict(ckpt['model'])
    if tool == 'vgg':
        # listen_list=['conv_1_2', 'conv_2_2', 'conv_3_2', 'conv_4_2',]
        #listen_list = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_3_2', 'conv_4_1']
        listen_list = ['conv_1_1', 'conv_2_1',] # 'conv_3_1', 'conv_3_2', 'conv_4_1'] #, 'conv_4_2',]
        #listen_list = [ 'conv_1_1', 'conv_2_1']#, 'conv_3_1', 'conv_3_2', 'conv_4_1']
        net = VGG_Model(load_ckpt_path=load_ckpt_path, listen_list=listen_list)
    if tool.startswith('flownet'):
        if tool == 'flownet':
            net = PoseFlowNet(3, 18, ngf=32, img_f=256, encoder_layer=5, attn_layer=[2,3], norm='instance', activation='LeakyReLU',
                                    use_spect=False, use_coord=False)   
        else:
            net = PoseFlowNet(3, 18, ngf=32, img_f=256, encoder_layer=4, attn_layer=[2], norm='instance', activation='LeakyReLU',
                                    use_spect=False, use_coord=False)
        # import pdb; pdb.set_trace()
        if load_ckpt_path:
            ckpt = torch.load(load_ckpt_path)
            net.load_state_dict(ckpt, strict=False)
            print("load ckpt from %s."%load_ckpt_path)
        else:
            return init_net(net, init_type, init_gain, gpu_ids)
    if tool == 'segmentor':
        net = Segmentor(
            img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64,
            n_human_parts=8, 
            n_downsampling=2, n_style_blocks=8, 
            norm_type='instance', relu_type='relu', 
        )
        if load_ckpt_path:
            ckpt = torch.load(load_ckpt_path)
            net.load_state_dict(ckpt)
            print("load ckpt from %s."%load_ckpt_path)
    print("[init] init pre-trained model %s." % tool)
    
    return init_net(net, gpu_ids=gpu_ids, do_init_weight=False)

def define_E(input_nc, output_nc, netE, ngf=64, n_downsample=3, norm_type='none', relu_type='relu', frozen_flownet=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if netE == 'adgan':
        net = ADGANEncoder(input_nc, output_nc, ngf=ngf, n_downsample=n_downsample, norm_type='none', relu_type=relu_type, frozen_flownet=frozen_flownet)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G(input_nc, output_nc, ngf, latent_nc, style_nc, n_downsampling=2, n_style_blocks=4, n_human_parts=8, netG='adgan', norm='instance', relu_type='relu', init_type='normal', init_gain=0.02, gpu_ids=[], **kwargs):
    Generator = find_generator_using_name(netG)
    # import pdb; pdb.set_trace()
    net = Generator(
            img_nc=3, kpt_nc=input_nc, ngf=ngf, latent_nc=latent_nc, style_nc=style_nc,
            n_human_parts=n_human_parts, 
            n_downsampling=n_downsampling, n_style_blocks=n_style_blocks, 
            norm_type=norm, relu_type=relu_type, **kwargs
            )
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_dropout=True, use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    #assert norm == "instance"
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'resnet':
        net = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3,
                                padding_type='reflect', use_sigmoid=use_sigmoid, n_downsampling=2)
    elif netD == 'gfla':
        net = ResDiscriminator(input_nc=input_nc, ndf=ndf, img_f=256, layers=n_layers_D, activation='LeakyReLU')
        return init_net(net, gpu_ids=gpu_ids, do_init_weight=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_shift_predictor(latent_nc, netSP, SP_size, init_type, init_gain, gpu_ids):
    if netSP == 'ResNet':
        shift_predictor = ResNetShiftPredictor(latent_nc, 8) #SP_size)
    elif netSP == 'LeNet':
        shift_predictor = LeNetShiftPredictor(latent_nc, 8)
    return init_net(shift_predictor, gpu_ids=gpu_ids, do_init_weight=False)

def define_deformator(latent_nc, netDeform, init_type, init_gain, gpu_ids):
    net = LatentDeformator(latent_nc, type=netDeform)
    return init_net(net, gpu_ids=gpu_ids, do_init_weight=False)
