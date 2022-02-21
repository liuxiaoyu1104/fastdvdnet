import cv2
import imageio
import numpy as np
import os
from utils.io import loop_until_success
from skimage.transform import resize

def crop_position(patch_size, H, W):
    position_h = np.random.randint(0, H-patch_size+1)
    position_w = np.random.randint(0, W-patch_size+1)
    return position_h, position_w

def list_dir(dir, postfix=None, full_path=False):
    if full_path:
        if postfix is None:
            names = sorted([name for name in os.listdir(dir) if not name.startswith('.')])
            return sorted([os.path.join(dir, name) for name in names])
        else:
            names = sorted([name for name in os.listdir(dir) if (not name.startswith('.') and name.endswith(postfix))])
            return sorted([os.path.join(dir, name) for name in names])
    else:
        if postfix is None:
            return sorted([name for name in os.listdir(dir) if not name.startswith('.')])
        else:
            return sorted([name for name in os.listdir(dir) if (not name.startswith('.') and name.endswith(postfix))])

def open_images_uint8(image_files):
    image_list = []
    for image_file in image_files:
#         image = imageio.imread(image_file).astype(np.float32)
        print(image_file)
        image = np.load(image_file).astype(np.float32)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image_list.append(image)
    seq = np.stack(image_list, axis=0)
    return seq
    
def open_images_scale(image_files,scale=1):
    image_list = []
    for image_file in image_files:
        image = np.load(image_file).astype(np.float32)
        h, w, c = image.shape
        image=np.flip(image,axis=2)
        image = cv2.resize(image, ( int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
        image=np.flip(image,axis=2)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image_list.append(image)
    seq = np.stack(image_list, axis=0)
    return seq
