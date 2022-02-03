"""
    :filename ConsensusMethodCENTROID.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Centroid consensus method calculates the mean value of predictions from several viewpoints.
"""

import numpy as np


class ConsensusCENTROID:
    """
    Centroid consensus method calculates the mean value of predictions from several viewpoints.
    """

    def __init__(self, estimates):
        self.estimates = estimates

    def estimate_CENTROID(self):
        """
        Calculate the mean position (Centroid) of N points in 3-space.

        Nx3-matrix containing points in 3-space from the method attribute is used.

        :return: tuple (x, y, z), which represents the Centroid of provided points.
        """

        return np.mean(self.estimates, axis=0)
