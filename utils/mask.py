import numpy as np
import cv2
import random
from scipy import ndimage, misc

class Masks:

    @staticmethod
    def get_ff_mask(h, w, num_v = None):
        #Source: Generative Inpainting https://github.com/JiahuiYu/generative_inpainting

        mask = np.zeros((h,w))
        if num_v is None:
            num_v = 15+np.random.randint(9) #5

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(4.0)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(60) # 40
                brush_w = 10+np.random.randint(15) # 10
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return mask.astype(np.float32)


    @staticmethod
    def get_box_mask(h,w):
        height, width = h, w

        mask = np.zeros((height, width))

        mask_width = random.randint(int(0.3 * width), int(0.7 * width)) 
        mask_height = random.randint(int(0.3 * height), int(0.7 * height))
 
        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)

        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    @staticmethod
    def get_ca_mask(h,w, scale = None, r = None):

        if scale is None:
            scale = random.choice([1,2,4,8])
        if r is None:
            r = random.randint(2,6) # repeat median filter r times

        height = h
        width = w
        mask = np.random.randint(2, size = (height//scale, width//scale))

        for _ in range(r):
            mask = ndimage.median_filter(mask, size=3, mode='constant')
        mask = cv2.resize(mask,(w,h), interpolation=cv2.INTER_NEAREST)
        # mask = transform.resize(mask, (h,w)) # misc.imresize(mask,(h,w),interp='nearest')
        if scale > 1:
            struct = ndimage.generate_binary_structure(2, 1)
            mask = ndimage.morphology.binary_dilation(mask, struct)


        return mask

    @staticmethod
    def get_random_mask(h,w):
        f = random.choice([Masks.get_box_mask, Masks.get_ca_mask, Masks.get_ff_mask]) 
        return f(h,w).astype(np.int32)
