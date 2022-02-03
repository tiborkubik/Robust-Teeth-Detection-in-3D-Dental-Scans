"""
    :filename RMSELoss.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    File for the RMSE loss function class.
"""

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error loss function class.

    This lf takes a square root of classic MSE loss.
    It implements its own forward method that is used in the training process.
    """
    def __init__(self, eps=1e-6):
        super().__init__()

        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self.eps)

        return loss
