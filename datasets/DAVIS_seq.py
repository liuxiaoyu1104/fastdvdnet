
import glob
import numpy as np
import os
import random
import torch
from torch.utils.data.dataset import Dataset
from utils.utils import open_sequence
import imageio
from skimage.transform import resize
import torch.nn.functional as F
from datasets.base_functions import crop_position, list_dir, open_images_scale

scales = [1, 0.9, 0.8, 0.7]

    

class Train_set(torch.utils.data.Dataset):
    def __init__(self, seq_dir, train_length, patch_size, patches_per_epoch, temp_stride=3, image_postfix='jpg'):
        self.seq_dir = seq_dir
        self.train_length = train_length
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.temp_stride = temp_stride

        self.seq_names = list_dir(seq_dir)
        self.seqs = {}
        for seq_name in self.seq_names:
            print(seq_name)
            self.seqs[seq_name] = {}
            self.seqs[seq_name]['clean_image_files'] = list_dir(os.path.join(self.seq_dir, seq_name),
                                                                postfix=image_postfix, full_path=True)
            for k in range(len(scales)):
                
                self.seqs[seq_name]['clean_images'+str(scales[k])] = open_images_scale(self.seqs[seq_name]['clean_image_files'],scales[k])

        self.seq_count = []
        for i in range(len(self.seq_names)):
            count = (self.seqs[self.seq_names[i]]['clean_images'+str(1)].shape[0] - self.train_length + self.temp_stride) // self.temp_stride
            self.seq_count.append(count)
        self.seq_count_cum = np.cumsum(self.seq_count)

    def __getitem__(self, index):
        if self.patches_per_epoch is not None:
            index = index % self.seq_count_cum[-1]
        for i in range(len(self.seq_count_cum)):
            if index < self.seq_count_cum[i]:
                seq_name = self.seq_names[i]
                seq_index = index if i == 0 else index - self.seq_count_cum[i - 1]
                break
        center_frame_index = seq_index * self.temp_stride + (self.train_length//2)
        aug = np.random.randint(0, 4)   
        data = self.seqs[seq_name]['clean_images'+str(scales[aug])][center_frame_index - (self.train_length//2):center_frame_index + (self.train_length//2) + (self.train_length%2)]

        # crop patchesa
        num_frames, C, H, W = data.shape
        position_H, position_W = crop_position(self.patch_size, H, W)
        data = data[:, :, position_H:position_H+self.patch_size, position_W:position_W+self.patch_size]

        return_dict = {'data':data}
        return return_dict

    def __len__(self):
        if self.patches_per_epoch is None:
            return self.seq_count_cum[-1]
        else:
            return self.patches_per_epoch

NUMFRXSEQ_VAL = 85    # number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class Val_set(Dataset):
    """Validation dataset. Loads all the images in the dataset folder on memory.
    """
    def __init__(self, valsetdir, gray_mode=True, num_input_frames=NUMFRXSEQ_VAL):
        self.gray_mode = gray_mode

        # Look for subdirs with individual sequences
        seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))

        # open individual sequences and append them to the sequence list
        sequences = []
        for seq_dir in seqs_dirs:
            seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
                             max_num_fr=num_input_frames)
            # seq is [num_frames, C, H, W]
            sequences.append(seq)

        self.seqs_dirs = seqs_dirs
        self.sequences = sequences

    def __getitem__(self, index):
        return {'seq':torch.from_numpy(self.sequences[index]), 'name':self.seqs_dirs[index]}

    def __len__(self):
        return len(self.sequences)