from functools import wraps
import cv2
import imageio
import numpy as np
import time
import torch

# this is for handling io errors
def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(10)
        return ret
    return wrapper

@loop_until_success
def log(log_file, str, also_print=True):
    with open(log_file, 'a+') as F:
        F.write(str)
    if also_print:
        print(str, end='')

# return pytorch image in shape 1x3xHxW
@loop_until_success
def image2tensor(image_file):
    image = imageio.imread(image_file).astype(np.float32) / np.float32(255.0)
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    elif len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    image = np.asarray(image, dtype=np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    return image

# save numpy image in shape 3xHxW
@loop_until_success
def np2image(image, image_file):
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0., 1.)
    image = image * 255.
    image = image.astype(np.uint8)
    imageio.imwrite(image_file, image)

def np2image_bgr(image, image_file):
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0., 1.)
    image = image * 255.
    image = image.astype(np.uint8)
    cv2.imwrite(image_file, image)

# save tensor image in shape 1x3xHxW
@loop_until_success
def tensor2image(image, image_file):
    image = image.detach().cpu().squeeze(0).numpy()
    np2image(image, image_file)

def np2flow_png(flow_np, flow_png_file):
    flow_png = flow_to_png_middlebury(flow_np)
    imageio.imwrite(flow_png_file, flow_png)

@loop_until_success
def tensor2flow_png(tensor, flow_png_file):
    flow = tensor.detach().cpu().squeeze(0).numpy()
    np2flow_png(flow, flow_png_file)

UNKNOWN_FLOW_THRESH = 1e7

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img

def flow_to_png_middlebury(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    flow = flow.transpose([1, 2, 0])
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)