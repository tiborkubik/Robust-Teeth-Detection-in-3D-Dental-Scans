"""
    :filename MissingTeethSampler.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    MissingTeethSampler class file.

    This class is a wrapper for PyTorch Sampler
    (https://pytorch.org/cppdocs/api/classtorch_1_1data_1_1samplers_1_1_sampler.html)
    and is used to provide more minority images in each batch.
"""

import torch
from torch.utils.data.sampler import Sampler


class MissingTeethSampler(Sampler):
    """
    Class samples the samples from dataset according to the number of samples of given class.
    It is used to over-sample the images that contain third molars, as these images are a minority class.
    No under-sampling or other over-sampling is applied.
    """
    def __init__(self, dataset):
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.indices = list(range(len(self.dataset)))

        teeth_distribution = {}

        print('Sampling imbalanced teeth 1 and 8...')
        for idx in self.indices:
            label = self.get_label(idx)
            if label in teeth_distribution:
                teeth_distribution[label] += 1
            else:
                teeth_distribution[label] = 1
            print(f'{idx}/{len(self.indices)}: {label}')

        teeth_distribution['1-present'] *= 16
        teeth_distribution['8-present'] *= 16
        teeth_distribution['1-and-8-present'] *= 16

        '''Calculate the weights for classes.'''
        weights = [1.0 / teeth_distribution[self.get_label(idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def get_label(self, idx):
        return self.dataset[idx]['label']

    def __iter__(self):
        return iter(torch.multinomial(self.weights, len(self.indices), replacement=True).tolist())

    def __len__(self):
        return len(self.indices)
