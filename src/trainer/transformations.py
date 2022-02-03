"""
    :filename transformations.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz
from
    Classes of custom transformations that are applied during the training as additional augmentation of the depth maps.
"""

import torch
import random
import numpy as np
import torch.nn.functional as F

from random import randrange
from skimage.transform import resize, warp, AffineTransform


class Normalize(object):
    """Normalization of a depth map in the value of [0, 1] for each pixel."""
    def __init__(self, input_type):
        self.input_type = input_type

    def __call__(self, sample):

        if self.input_type == 'geom':
            image, landmarks, label = sample['image'], sample['landmarks'], sample['label']

            mean, std = image.mean([1, 2]), image.std([1, 2])

            # TODO?

            return {'image': image,
                    'landmarks': landmarks,
                    'label': label}


class ToTensor(object):
    """Transformation of a training sample into a torch tensor instance."""
    def __init__(self, input_type):
        self.input_type = input_type

    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']

        image = torch.from_numpy(image.copy())

        if self.input_type != 'depth+geom':
            image = image.unsqueeze(1)
            image = image.permute(1, 0, 2)
        else:
            image = image.permute(2, 0, 1)

        landmarks = np.asarray(landmarks)
        landmarks = torch.from_numpy(landmarks.copy())

        return {'image': image,
                'landmarks': landmarks,
                'label': label}


class Resize(object):
    """Resizing of the input sample into provided dimensions."""

    def __init__(self, width, height, input_type='image'):
        assert isinstance(width, int)
        assert isinstance(height, int)

        self.width = width
        self.height = height
        self.type = input_type

    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        resized_landmarks = landmarks.copy()

        if self.type == 'image':
            image = resize(image, (self.height, self.width), anti_aliasing=True)
        if self.type == 'landmarks':
            resized_landmarks = []
            for landmark in landmarks:
                landmark_resized = resize(landmark, (self.height, self.width), anti_aliasing=True)
                resized_landmarks.append(landmark_resized)

        return {'image': image,
                'landmarks': resized_landmarks,
                'label': label}


class RandomTranslating(object):
    """Randomly translate the input sample from range [-10 px, 10 px] with provided probability."""

    def __init__(self, p=0.5):
        assert isinstance(p, float)

        self.p = p

    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        translated_landmarks = landmarks.copy()

        if np.random.rand(1) < self.p:
            n1 = randrange(-10, 10)
            n2 = randrange(-10, 10)

            t = AffineTransform(translation=(n1, n2))

            image = warp(image, t.inverse)

            translated_landmarks = []
            for landmark in landmarks:
                translated_landmarks.append(warp(landmark, t.inverse))

        return {'image': image,
                'landmarks': translated_landmarks,
                'label': label}


class RandomScaling(object):
    """Randomly scales the input sample with scale index from range [0.90, 1.10] with provided probability."""

    def __init__(self, p=0.5):
        assert isinstance(p, float)

        self.p = p

    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        scaled_landmarks = landmarks.copy()

        if np.random.rand(1) < self.p:
            n = random.uniform(0.90, 1.10)
            t = AffineTransform(scale=(n, n))

            image = warp(image, t.inverse)

            scaled_landmarks = []
            for landmark in landmarks:
                scaled_landmarks.append(warp(landmark, t.inverse))

        return {'image': image,
                'landmarks': scaled_landmarks,
                'label': label}


class RandomRotation(object):
    """Randomly rotates the input sample from range [−11.25 deg, 11.25 deg] with provided probability."""

    def __init__(self, p=0.5):
        assert isinstance(p, float)

        self.p = p

    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']

        rnd_num1 = randrange(-32, -6)
        rnd_num2 = randrange(6, 32)
        rnd_num = random.choice([rnd_num1, rnd_num2])

        if np.random.rand(1) < self.p:
            rotated_image = self.rotate(x=image.unsqueeze(0).type(torch.FloatTensor), theta=np.pi/rnd_num)

            rotated_landmarks = []
            for _, landmark in enumerate(landmarks):
                rotated_landmark = self.rotate(x=landmark.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor), theta=np.pi/rnd_num)
                rotated_landmarks.append(rotated_landmark.squeeze(0))

            result = torch.cat(rotated_landmarks, dim=0)

            return {'image': rotated_image.squeeze(0),
                    'landmarks': result,
                    'label': label}

        return {'image': image,
                'landmarks': landmarks,
                'label': label}

    @staticmethod
    def get_rotation_matrix(theta):
        """Returns a tensor rotation matrix with given theta value."""

        theta = torch.tensor(theta)

        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def rotate(self, x, theta):
        rot_mat = self.get_rotation_matrix(theta)[None, ...].repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x
