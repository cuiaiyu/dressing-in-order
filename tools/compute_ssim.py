"""
This file is build upon PATN at https://github.com/tengteng95/Pose-Transfer/blob/master/tool/getMetrics_fashion.py.
"""
import os
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.util.arraycrop import crop
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
import re
import cv2
from PIL import Image



def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):

        #import pdb; pdb.set_trace()
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    #import pdb; pdb.set_trace()
    print('ssim: %f' % np.mean(ssim_score_list))
    return np.mean(ssim_score_list)



def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images



def addBounding(image, shrink_rate=1, sharp_rate=1, bound=40):

    #image = sharpen(image, sharp_rate)
    #image = shrink(image, shrink_rate)
    # import pdb; pdb.set_trace()
    # image = cv2.resize(image, (176, 256))
    # return image
    h, w, c = image.shape
    if h == w:
        return image
    image_bound = np.ones((h, w+bound*2, c))*255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound+w] = image

    return image_bound

def shrink(image, shrink_rate):
    h,w,c = image.shape
    new_w = int(shrink_rate * w)
    image = cv2.resize(image, (new_w, 256))
    image = cv2.resize(image, (176, 256))
    return image

def sharpen(img, alpha=1):
    if alpha == 1:
        return img
    elif alpha == 2:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    elif alpha == 3:
        #Create the identity filter, but with the 1 shifted to the right!
        kernel = np.zeros( (9,9), np.float32)
        kernel[4,4] = 2.0   #Identity, times two! 
        #Create a box filter:
        boxFilter = np.ones( (9,9), np.float32) / 81.0
        #Subtract the two:
        kernel = kernel - boxFilter
    elif alpha == 4:
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    elif alpha == 5:
        kernel = np.ones((3,3)) / 9.0
    elif alpha == 6:
        kernel = np.array([[1,2,1], [2,4,2], [1,2,1]])
    

    img = cv2.filter2D(img, -1, kernel)
    return img

def load_generated_images(images_folder, shrink_rate=1, sharp_rate=1, w=176):
    input_images = []
    target_images = []
    generated_images = []

    names = []
    print("image width: %d" % w)
    for img_name in os.listdir(images_folder):
        # import pdb; pdb.set_trace()
        if not (img_name.endswith("jpg") or img_name.endswith("png")):
            continue
        img = imread(os.path.join(images_folder, img_name))
        
        # import pdb; pdb.set_trace()
        # w = 176
        input_images.append(addBounding(img[:, :w]))
        target_images.append(addBounding(img[:, w:2*w]))
        generated_images.append(addBounding(img[:, 2*w:], shrink_rate, sharp_rate))
    return input_images, target_images, generated_images


def test(generated_images_dir, masks=None, shrink_rate=1, sharp_rate=1,w=176):
    # load images
    print ("Loading images...")
    input_images, target_images, generated_images = load_generated_images(generated_images_dir, 
    shrink_rate=shrink_rate, sharp_rate=sharp_rate, w=w)
    print("[log] %d images loaded" % len(input_images))
    print ("Computing SSIM...")
    ssim= ssim_score(generated_images, target_images)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SSIM Eval.')
    parser.add_argument('--output_dir', type=str, help='an integer for the accumulator')
    parser.add_argument('--square', action='store_true', help='is square image. (256x256)')
    opt = parser.parse_args()

    print("load generated image from %s" % opt.output_dir)
    # SSIM
    test(opt.output_dir, w=256 if opt.square else 176)
    