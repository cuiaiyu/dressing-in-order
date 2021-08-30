import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from tensorboardX import SummaryWriter
import shutil

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:%d'%self.gpu_ids[0]) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        
        self.loss_names = []
        self.model_names = []
        self.frozen_models = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.loss_coe = {}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # frozen model
        for model in self.frozen_models:
            model = 'net' + model
            net = self
            for m in model.split('.'):
                net = getattr(net, m)
            self.set_requires_grad(net, False)
            net.eval()
            print("[init] frozen net %s." % model)
            
        # optimizer
        self._init_optimizer(opt)
        
        # scheduler
        #if self.isTrain:
        
        epoch_count = -1
        
        # load model
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            epoch_count = self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
        
        # tensorboard, log
        self._init_tensorboard(opt)
        self.log_loss_update(reset=True)    
        
        return epoch_count
    
    def _init_optimizer(self, opt):
        if self.isTrain:
            for model in self.frozen_models:
                model = 'net' + model
                net = self
                for m in model.split('.'):
                    net = getattr(net, m)
                self.set_requires_grad(net, False) 

            G_params, D_params = [], []
            G_names, D_names = [], []
            for name in self.model_names:
                if name in self.frozen_models:
                    continue
                elif name.startswith("D"):
                    D_params += [param for param in getattr(self, "net"+name).parameters() if param.requires_grad]
                    D_names.append(name)
                else:
                    G_params += [param for param in getattr(self, "net"+name).parameters() if param.requires_grad]
                    G_names.append(name)
            if G_params:
                self.optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                print("[optimizer G]: %s" % ", ".join(G_names))
            if D_params:
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr*opt.g2d_ratio, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
                print("[optimizer D]: %s" % ", ".join(D_names))
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
            
                
    def _init_tensorboard(self, opt):
        tb_name = "train" if opt.isTrain else "test"
        self.tb_dir = os.path.join(self.save_dir, tb_name)
        if not self.isTrain or not opt.continue_train:
            if os.path.exists(self.tb_dir):
                shutil.rmtree(self.tb_dir)
            os.mkdir(self.tb_dir)
        self.image_dir = os.path.join(self.tb_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir) 
        print("[tensorboard] init tensorboard @ %s" % self.tb_dir)
        
    def eval(self):
        """Make models eval mode during test time"""
        self.isTrain = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
                
    def train(self):
        """Make models eval mode during test time"""
        self.isTrain = True
        for name in self.model_names:
            if isinstance(name, str) and not name in self.frozen_models:
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            
    def log_loss_update(self, reset=False):
        # reset if required
        if reset:
            for name in self.loss_names:
                cum_loss_name = 'cum_loss_' + name
                setattr(self, cum_loss_name, 0.0)
                self.print_cnt = 0
            return
        # update
        for name in self.loss_names:
            cum_loss_name = 'cum_loss_' + name
            loss_name = 'loss_' + name
            if not hasattr(self, loss_name):
                continue
            cum_loss = getattr(self, cum_loss_name)
            curr_loss = getattr(self, loss_name)
            if isinstance(curr_loss, torch.Tensor):
                curr_loss = curr_loss.item()
            setattr(self, cum_loss_name, cum_loss + curr_loss)
        self.print_cnt += 1
            
    def get_cum_losses(self):
        ret = {}
        for name in self.loss_names:
            cum_loss_name = 'cum_loss_' + name
            cum_loss = getattr(self, cum_loss_name)
            ret[name] = cum_loss / self.print_cnt
        return ret
            
    def compute_visuals(self, step, loss_only=False,name=''):
        if not loss_only:
            print_img = []
            for v in self.visual_names:
                print_img.append(getattr(self, v).float().cpu().detach())
            print_img = torch.cat(print_img, 2)
            print_img = (print_img + 1) / 2.0
            self.writer.add_images('examples' + name, print_img[:8], step)

        # numbers
        if self.isTrain:
            losses = self.get_cum_losses()
            self.log_loss_update(reset=True)
            for loss_name in losses:
                self.writer.add_scalar(loss_name, losses[loss_name], step)

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.2e' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, iter_count=-1):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            #if name == 'VGG':
            #   continue
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
        if epoch == "latest":
            save_path = os.path.join(self.save_dir, "latest_iter.txt")
            with open(save_path,'w') as f:
                f.write("%d"%iter_count)


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            #if name == 'VGG':
            #    continue
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                if not os.path.exists(load_path):
                    print("not exsits %s" % load_path)
                    continue
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict, strict=False)
                
        if epoch == 'latest':
            load_path = os.path.join(self.save_dir, "latest_iter.txt")
            if os.path.exists(load_path):
                with open(load_path) as f:
                    epoch_count = f.readline()
                return int(epoch_count)
        return -1

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad