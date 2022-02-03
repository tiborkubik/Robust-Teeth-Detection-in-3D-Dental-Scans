import math
import numpy as np

from src import config


def find_intersection_of_N_lines(p_a, p_b):
    """
    Find intersection point of lines in 3D space, in the least squares sense.

    Algorithm inspiration from: https://se.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of
    -lines-in-3d-space?focused=5235003&tab=function

    Original author:
    Anders Eikenes, 2012

    License:
    Copyright (c) 2012, Anders Eikenes
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :param p_a: Nx3-matrix containing starting point of N lines.
    :param p_b: Nx3-matrix containing end point of N lines.

    :return: Best intersection point of the N lines, in least squares sense.
    """

    lines_num = p_a.shape[0]

    '''Lines description in a form of vectors.'''
    s_i = p_b - p_a

    '''Normalisation of vectors.'''
    n_i = np.divide(s_i, np.transpose(np.sqrt(np.sum(s_i ** 2, 1)) * np.ones((3, lines_num))))
    n_x = n_i[:, 0]
    n_y = n_i[:, 1]
    n_z = n_i[:, 2]

    xx = np.sum(n_x ** 2 - 1)
    yy = np.sum(n_y ** 2 - 1)
    zz = np.sum(n_z ** 2 - 1)

    xy = np.sum(np.multiply(n_x, n_y))
    xz = np.sum(np.multiply(n_x, n_z))
    yz = np.sum(np.multiply(n_y, n_z))

    s = np.array([[xx, xy, xz],
                  [xy, yy, yz],
                  [xz, yz, zz]])

    cx = np.sum(np.multiply(p_a[:, 0], (n_x ** 2 - 1)) +
                np.multiply(p_a[:, 1], np.multiply(n_x, n_y)) +
                np.multiply(p_a[:, 2], np.multiply(n_x, n_z)))
    cy = np.sum(np.multiply(p_a[:, 0], np.multiply(n_x, n_y)) +
                np.multiply(p_a[:, 1], (n_y ** 2 - 1)) +
                np.multiply(p_a[:, 2], np.multiply(n_y, n_z)))
    cz = np.sum(np.multiply(p_a[:, 0], np.multiply(n_x, n_z)) +
                np.multiply(p_a[:, 1], np.multiply(n_y, n_z)) +
                np.multiply(p_a[:, 2], (n_z ** 2 - 1)))

    p_intersect = np.matmul(np.linalg.pinv(s), np.array([[cx], [cy], [cz]]))  # pinv(s) is the Moore-Penrose pseudo inv

    return p_intersect[:, 0]


class GeometricConsensusRANSAC:
    """
    Finding the point estimate in the multi-view manner using a geometric approach based on RANSAC
    and least-squares fit algorithms.

    The idea was presented in following paper: https://arxiv.org/abs/1910.06007.

    Please refer also to corresponding github repository, the core of this algorithm is inspired by their method and
    modified according to my needs:

    Link: https://github.com/RasmusRPaulsen/Deep-MVLM/blob/d0240f567deecef9ea70c2e5690fcdb62f29f330/utils3d/utils3d.py#L225

    Licence:
    MIT License

    Copyright (c) 2019 Rasmus R. Paulsen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, p_a, p_b):
        self.p_a = p_a
        self.p_b = p_b

        self.best_pos = (math.inf, math.inf, math.inf)
        self.best_err = math.inf

        self.pred_inliers = 0
        self.pred_outliers = 0

        # print(self.n_of_lines)

    @property
    def n_of_lines(self):
        return len(self.p_a)

    @property
    def threshold(self):
        """One quarter of all lines."""
        return self.n_of_lines/3

    def point_line_distance(self, p, idx=None):
        """Calculates the point-line distance. https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html."""
        if idx is None:
            p_a, p_b = self.p_a, self.p_b
        else:
            p_a, p_b = self.p_a[idx, :], self.p_b[idx, :]

        return np.cross((np.transpose(p) - p_a),
                        (np.transpose(p) - p_b))

    def sum_squared_distances(self, A, B, num_of_inliers):
        distances = self.get_distances(A, B)

        return np.sum(np.square(distances))/num_of_inliers

    def estimate_RANSAC(self):
        for _ in range(config.HYPERPARAMETERS['ransac_iterations']):
            rnd_lines = np.random.choice(range(self.n_of_lines), 3, replace=False)

            if self.p_a is not None and self.p_b is not None:
                '''Find the initial estimate from three lines.'''
                point_estimate = find_intersection_of_N_lines(self.p_a[rnd_lines, :], self.p_b[rnd_lines, :])

                A = self.point_line_distance(point_estimate)
                B = self.p_b - self.p_a
                distances = self.get_distances(A, B)

                '''Calculate the number of inliers according to the tolerance.'''
                n_inliers = np.sum(distances < config.HYPERPARAMETERS['ransac_threshold_mm'])

                '''If the number of inliers is greater than the allowed threshold:'''
                if n_inliers > self.threshold:
                    '''Reestimate based on the inliers'''
                    idx = distances < config.HYPERPARAMETERS['ransac_threshold_mm']
                    point_estimate = find_intersection_of_N_lines(self.p_a[idx, :], self.p_b[idx, :])

                    '''Compute distance from all inlines to intersection.'''
                    A = self.point_line_distance(point_estimate, idx)
                    B = self.p_b[idx, :] - self.p_a[idx, :]

                    sum_squared = self.sum_squared_distances(A, B, n_inliers)
                    if sum_squared < self.best_err:
                        self.best_err = sum_squared
                        self.best_pos = point_estimate

        # self.pred_inliers = n_inliers
        # self.pred_outliers = self.n_of_lines - n_inliers

        # ratio = self.pred_inliers/self.pred_outliers if self.pred_outliers != 0 else 1

        # print(f'RANSAC prediction finished. Number of inliers: {self.pred_inliers}.'
        #       f'Number of outliers: {self.pred_outliers}.'
        #       f'Inlier/outlier ratio: {ratio}')

        return self.best_pos

    @staticmethod
    def get_distances(A, B):
        return (np.linalg.norm(A, axis=1)/np.linalg.norm(B, axis=1)) ** 2
