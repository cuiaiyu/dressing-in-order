"""
This file is originally from GFLA at https://github.com/RenYurui/Global-Flow-Local-Attention/blob/master/script/generate_fashion_datasets.py
This is a modified version updated by Aiyu Cui.

Run as
python generate_fashion_datasets.py --dataroot $DATAROOT
"""
import os
import shutil
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_anno(anno_path):
    train_images = []
    train_f = open(anno_path, 'r')
    for lines in train_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            train_images.append(lines[:-4])
    return train_images

def convert_file_name(catagory, fn):
    pass
    

def make_dataset(dataroot):
    assert os.path.isdir(dataroot), '%s is not a valid directory' % dataroot
    
    # load data split from annotaton file
    train_root = os.path.join(dataroot, 'train')
    if not os.path.exists(train_root):
        os.mkdir(train_root)
        
    test_root = os.path.join(dataroot, 'test')
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    train_images = load_anno(os.path.join(dataroot, 'train.lst'))
    test_images =  load_anno(os.path.join(dataroot, 'test.lst'))
    
    # split data
    img_root = os.path.join(dataroot, 'img_highres')
    for root, _, fnames in sorted(os.walk(img_root)):
        for fname in fnames:
            if not is_image_file(fname):
                continue
            path = os.path.join(root, fname)
            print("Load Image", path)
            path_names = path.split('/')
            path_names = path_names[len(img_root.split("/")):]
            path_names = ['fashion'] + path_names
            path_names[3] = path_names[3].replace('_', '')
            path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
            path_names = "".join(path_names)

            if path_names[:-4] in train_images:
                shutil.copy(path, os.path.join(train_root, path_names))
                print("Save to", os.path.join(train_root, path_names))

            elif path_names[:-4] in test_images:
                shutil.copy(path, os.path.join(test_root, path_names))
                print("Save to", os.path.join(train_root, path_names))
                #pass
                
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataroot', type=str, default="data", help='data root')

    args = parser.parse_args()
    
    make_dataset(args.dataroot)