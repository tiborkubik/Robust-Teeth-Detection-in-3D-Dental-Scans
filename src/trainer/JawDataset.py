"""
    :filename JawDataset.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    JawDataset class file.

    Dataset wrappers used for the data representation during the training procedure and evaluation.
"""

import os
import math
import glob
import torch
import numpy as np

from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset

import config


class JawDataset(Dataset):
    """Dataset class used for training and validation."""

    def __init__(self, dataset_path, transform=None, input_type='depth'):
        self.landmarks = []

        self.dataset_path = dataset_path
        self.transform = transform
        self.input_type = input_type

        assert self.input_type in ['depth', 'geom', 'depth+geom']

        EXT = "*.csv"

        all_csv_files = [file
                         for path, subdir, files in os.walk(self.dataset_path)
                         for file in glob.glob(os.path.join(path, EXT))]

        for csv_name in all_csv_files:
            lms = np.genfromtxt(csv_name, delimiter=',', usecols=(0, 1, 2), skip_header=1)
            lms = lms[~np.isnan(lms).any(axis=1)]

            self.landmarks.append([lms, csv_name])

        print(len(self.landmarks))
        self.landmarks = self.landmarks[:2000]  # for dataset split...

    def __getitem__(self, index):
        img_path = str(self.landmarks[index][-1]).replace('landmarks', 'images')

        if self.input_type == 'depth':
            img_path = img_path[:-3]
            img_path = img_path + 'png'

            img = Image.open(img_path)
        elif self.input_type == 'geom':
            view_num = img_path[-6:-4]

            img_path = img_path[:-6]
            img_path = img_path + 'GEOM_' + view_num + '.png'

            img = Image.open(img_path)
            img = img.convert('L')
        else:
            # depth
            img_path1 = img_path[:-3]
            img_path1 = img_path1 + 'png'

            img_depth = Image.open(img_path1)
            img_depth = np.array(img_depth)

            # geom
            view_num = img_path[-6:-4]

            img_path2 = img_path[:-6]
            img_path2 = img_path2 + 'GEOM_' + view_num + '.png'

            img_geom = Image.open(img_path2)
            img_geom = img_geom.convert('L')
            img_geom = np.array(img_geom)

            img = np.zeros((config.DIMENSIONS['original'], config.DIMENSIONS['original'], 2), dtype=np.float32)

            img[:, :, 0] = img_depth[:, :]
            img[:, :, 1] = img_geom[:, :]

        img = np.array(img)
        img = img / 255  # [0, 1]
        # img = img[::-1, :]  # revise height in (height, width, channel)
        img = img.astype(float)

        # plt.figure()
        # show_landmarks(img, pd.read_csv(self.landmarks[index][-1]))
        # plt.show()

        heatmaps_dir_path = str(self.landmarks[index][-1][:-4]) + '/'

        heatmaps = []

        '''Labels are used just for the over-sampling, otherwise no need for them.'''
        if 10.0 in self.landmarks[index][0] and 160.0 in self.landmarks[index][0]:
            label = '1-and-8-present'
        elif 10.0 in self.landmarks[index][0]:
            label = '1-present'
        elif 160.0 in self.landmarks[index][0]:
            label = '8-present'
        else:
            label = 'missing-both'

        for notation in config.VALID_NOTATIONS:
            heatmap = Image.open(heatmaps_dir_path + str(notation) + '.png')
            heatmap = np.array(heatmap)

            # heatmap = resize(heatmap, (32, 32), anti_aliasing=True)
            # plt.imshow(heatmap)
            # plt.show()

            # plt.imshow(heatmap)
            # plt.show()

            heatmap = heatmap / 255  # [0, 1]
            heatmaps.append(heatmap)

        sample = {'image': img,
                  'landmarks': heatmaps,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.landmarks)


class EvalJawDataset(Dataset):
    """Dataset for loading the outputs of rendering pipeline in evaluation of unseen polygons."""

    def __init__(self, net_input_renders, input_format='depth'):
        self.net_input_renders = net_input_renders
        self.input_format = input_format

    def __getitem__(self, index):
        if self.input_format == 'depth':
            image = resize(self.net_input_renders[index]['depth_map'], (128, 128), anti_aliasing=True)
        elif self.input_format == 'geom':
            image = resize(self.net_input_renders[index]['geometry'], (128, 128), anti_aliasing=True)
            image = self.rgb2gray(image)

        else:
            img_depth = resize(self.net_input_renders[index]['depth_map'], (128, 128), anti_aliasing=True)
            img_geom = resize(self.net_input_renders[index]['geometry'], (128, 128), anti_aliasing=True)
            img_geom = self.rgb2gray(img_geom)

            image = np.zeros((128, 128, 2), dtype=np.float32)

            image[:, :, 0] = img_depth[:, :, 0]  # img.depth.shape: (128, 128, 1)
            image[:, :, 1] = img_geom[:, :]  # img_geom.shape: (128, 128)

        image = image.astype(float)

        image = torch.from_numpy(image.copy())

        if self.input_format == 'geom':
            image = image.unsqueeze(2)
        image = image.permute(2, 0, 1)

        return {'image': image,
                'camera': self.net_input_renders[index]['camera']}

    def __len__(self):
        return len(self.net_input_renders)

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
