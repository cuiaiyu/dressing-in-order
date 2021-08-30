"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.test_options import TestOptions
from datasets import create_dataset
from models import create_model
import os, torch, shutil
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    if opt.square:
        opt.crop_size = (opt.crop_size, opt.crop_size)
    else:
        opt.crop_size = (opt.crop_size, max(1,int(opt.crop_size*1.0/256*176)))
    print("crop_size:", opt.crop_size)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    model.eval()
    
    generate_out_dir = os.path.join(opt.eval_output_dir + "_%s"%opt.epoch)
    print("generate images at %s" % generate_out_dir)
    os.mkdir(generate_out_dir)
    model.isTrain = False
    # generate
    count = 0
    for i, data in tqdm(enumerate(dataset), "generating for test split"):  # inner loop within one epoch
        with torch.no_grad():
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.forward()
            count = model.save_batch(generate_out_dir, count)
        
    
