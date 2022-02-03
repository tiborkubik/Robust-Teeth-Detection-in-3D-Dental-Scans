"""
    :filename CBRMSELoss.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    File for the class-balanced RMSE loss function class.
"""
import math
import statistics
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable

from src import config


class CBRMSELoss(nn.Module):
    """
    Root Mean Squared Error loss function class.

    This lf takes a square root of classic MSE loss.
    It implements its own forward method that is used in the training process.
    """

    def __init__(self, eps=1e-6):
        super().__init__()

        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y, labels):
        weights = np.array([self.weight_map(xi) for xi in labels])
        weights = Variable(torch.Tensor(weights)).cuda()

        loss = torch.sqrt(self.mse(x, y) + self.eps)
        loss = torch.mul(weights, loss)

        return loss.mean()

    def weight_map(self, x):
        if x == '1-present':
            return self.get_weight(class_count=config.TRAIN_SET_LM_COUNT[1])

        if x == '8-present':
            return self.get_weight(class_count=config.TRAIN_SET_LM_COUNT[31])

        if x == '1-and-8-present':
            w_1 = self.get_weight(class_count=config.TRAIN_SET_LM_COUNT[1])
            w_8 = self.get_weight(class_count=config.TRAIN_SET_LM_COUNT[31])

            return statistics.mean([w_1, w_8])

        return 1.17967  # Pre-calculated weight value for class balancing

    @staticmethod
    def get_weight(class_count):

        return ((1 - config.CLASS_BALANCE_BETA) /
                (1 - math.pow(config.CLASS_BALANCE_BETA, class_count))) * config.TOTAL_TRAIN_SET_COUNT

    # @staticmethod
    # def weighted_mse_loss(input, target, weight):
    #     return (weight * (input - target) ** 2).mean()
