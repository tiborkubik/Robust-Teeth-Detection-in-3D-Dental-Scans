"""
    :filename PerformanceMeasure.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    PerformanceMeasure class file.

    This class yields all values needed for the evaluation of landmark placement accuracy, and
    landmark presence accuracy.

    This class can measure:
    (1) Metrics for the landmark detection accuracy:
        (i)     Mean radial error for CENTROID,
        (ii)    Mean radial error for RANSAC,
        (iii)   Standard deviation for CENTROID,
        (iv)    Standard deviation for RANSAC,
        (v)     Overall SRDs for CENTROID (2 mm, 2.5 mm, 4 mm),
        (vi)    Overall SDRS for RANSAC (2 mm, 2.5 mm, 4 mm).
        + plot box plots and overall SDRs for individual landmarks

    (2) Metrics for the landmark presence:
        (i)     TP, TN, FP, and FN values
        (ii)    Precision,
        (iii)   Negative predictions value,
        (iv)    Accuracy,
        (v)     Specificity,
        (vi)    Sensitivity.
"""

import glob
import numpy as np
# import matplotlib.pyplot as plt
import random
import matplotlib

from collections import Counter

import config

matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': "pdflatex",
    'font.family': 'serif',
    'font.size': 16,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


class PerformanceMeasure:
    """Class for storing the measured values of individual landmarks on evaluated polygonal models."""

    def __init__(self, test_set_len):
        self.test_set_len = test_set_len

        self.distances_RANSAC = np.empty(shape=(self.test_set_len, config.LANDMARKS_NUM))
        self.distances_RANSAC[:] = np.nan
        self.distances_CENTROID = np.empty(shape=(self.test_set_len, config.LANDMARKS_NUM))
        self.distances_CENTROID[:] = np.nan

        self.heatmap_certainties_vals_present = []
        self.heatmap_certainties_vals_missing = []

        self.heatmap_certainties_preds = []  # always tuple, e.g [True, True] -> [is missing pred, GT]

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

        self.TP_3D = 0
        self.FP_3D = 0
        self.FN_3D = 0
        self.TN_3D = 0

    @property
    def lds_absence_precision(self):
        try:
            return round(self.TP/(self.TP + self.FP), 4) * 100
        except ZeroDivisionError:
            return -1

    @property
    def lds_absence_negative_pred_val(self):
        try:
            return round(self.TN/(self.TN + self.FN), 4) * 100
        except ZeroDivisionError:
            return -1

    @property
    def lds_absence_accuracy(self):
        try:
            return round((self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN), 4) * 100
        except ZeroDivisionError:
            return -1

    @property
    def lds_absence_accuracy_3D(self):
        return round((self.TP_3D + self.TN_3D) / (self.TP_3D + self.TN_3D + self.FP_3D + self.FN_3D), 4) * 100

    @property
    def lds_absence_specificity(self):
        try:
            return round(self.TN/(self.TN + self.FP), 4) * 100
        except ZeroDivisionError:
            return -1

    @property
    def lds_absence_sensitivity(self):
        try:
            return round(self.TP/(self.TP + self.FN), 4) * 100
        except ZeroDivisionError:
            return -1

    @property
    def mean_radial_error_RANSAC(self):
        mask = ~np.isnan(self.distances_RANSAC)
        filtered_data = [d[m] for d, m in zip(self.distances_RANSAC.T, mask.T)]

        N = 0
        sum_r = 0.0

        for f in filtered_data:
            sum_r += np.sum(f)
            N += len(f)

        if N != 0:
            R = sum_r/N
        else:
            R = -1

        return R

    @property
    def mean_radial_error_CENTROID(self):
        mask = ~np.isnan(self.distances_CENTROID)
        filtered_data = [d[m] for d, m in zip(self.distances_CENTROID.T, mask.T)]

        N = 0
        sum_r = 0.0

        for f in filtered_data:
            sum_r += np.sum(f)
            N += len(f)

        if N != 0:
            R = sum_r / N
        else:
            R = -1

        return R

    @property
    def SD_RANSAC(self):
        """Standard deviation for RANSAC consensus method."""
        mask = ~np.isnan(self.distances_RANSAC)
        filtered_data = [d[m] for d, m in zip(self.distances_RANSAC.T, mask.T)]

        vals = []

        for f in filtered_data:
            for val in f:
                vals.append(val)

        SD = np.std(np.asarray(vals), dtype=np.float64)

        return SD

    @property
    def SD_CENTROID(self):
        """Standard deviation for CENTROID consensus method."""
        mask = ~np.isnan(self.distances_CENTROID)
        filtered_data = [d[m] for d, m in zip(self.distances_CENTROID.T, mask.T)]

        vals = []

        for f in filtered_data:
            for val in f:
                vals.append(val)

        SD = np.std(np.asarray(vals), dtype=np.float64)

        return SD

    @property
    def overall_SDR_RANSAC(self):
        """Measures the overall SDR value at acceptance values 2 mm, 2.5 mm, and 4 mm for the RANSAC consensus."""

        lms_under_2 = 0
        lms_under_2_5 = 0
        lms_under_4 = 0
        lms_num = 0

        mask = ~np.isnan(self.distances_RANSAC)
        filtered_data = [d[m] for d, m in zip(self.distances_RANSAC.T, mask.T)]

        for f in filtered_data:
            for lm in f:
                if lm <= 2.0:
                    lms_under_2 += 1
                if lm <= 2.5:
                    lms_under_2_5 += 1
                if lm <= 4.0:
                    lms_under_4 += 1

            lms_num += len(f)

        percentage_2 = 0
        percentage_2_5 = 0
        percentage_4 = 0

        if lms_num != 0:
            percentage_2 = round((lms_under_2 / lms_num) * 100, 2)
            percentage_2_5 = round((lms_under_2_5 / lms_num) * 100, 2)
            percentage_4 = round((lms_under_4 / lms_num) * 100, 2)

        return percentage_2, percentage_2_5, percentage_4

    @property
    def overall_SDR_CENTROID(self):
        """Measures the overall SDR value at acceptance values 2 mm, 2.5 mm, and 4 mm for the CENTROID consensus."""

        lms_under_2 = 0
        lms_under_2_5 = 0
        lms_under_4 = 0
        lms_num = 0

        mask = ~np.isnan(self.distances_CENTROID)
        filtered_data = [d[m] for d, m in zip(self.distances_CENTROID.T, mask.T)]

        for f in filtered_data:
            for lm in f:
                if lm <= 2.0:
                    lms_under_2 += 1
                if lm <= 2.5:
                    lms_under_2_5 += 1
                if lm <= 4.0:
                    lms_under_4 += 1

            lms_num += len(f)

        percentage_2 = 0
        percentage_2_5 = 0
        percentage_4 = 0

        if lms_num != 0:
            percentage_2 = round((lms_under_2 / lms_num) * 100, 2)
            percentage_2_5 = round((lms_under_2_5 / lms_num) * 100, 2)
            percentage_4 = round((lms_under_4 / lms_num) * 100, 2)

        return percentage_2, percentage_2_5, percentage_4

    def certainty_metrics_calc(self):
        for p in self.heatmap_certainties_preds:
            if p[0] is True and p[1] is True:
                self.TP += 1

            if p[0] is True and p[1] is False:
                self.FP += 1

            if p[0] is False and p[1] is True:
                self.FN += 1

            if p[0] is False and p[1] is False:
                self.TN += 1

    def get_distances_RANSAC(self):
        return self.distances_RANSAC

    def get_distances_CENTROID(self):
        return self.distances_CENTROID

    def append_RANSAC(self, model_id, ld_id, value):
        self.distances_RANSAC[model_id][ld_id] = value

    def append_CENTROID(self, model_id, ld_id, value):
        self.distances_CENTROID[model_id][ld_id] = value

    def get_box_plots_each_landmark(self, consensus='RANSAC'):
        """Method plots the radial errors at each landmark """

        assert consensus in ['RANSAC', 'CENTROID']

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        ax.axhline(2, linestyle='--', color='r', label='Acceptable distance (2 mm)')

        if consensus == 'RANSAC':
            mask = ~np.isnan(self.distances_RANSAC)
            filtered_data = [d[m] for d, m in zip(self.distances_RANSAC.T, mask.T)]
            bp = ax.boxplot(filtered_data)
        else:
            mask = ~np.isnan(self.distances_CENTROID)
            filtered_data = [d[m] for d, m in zip(self.distances_CENTROID.T, mask.T)]
            bp = ax.boxplot(filtered_data)

        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(0, 32)])
        ax.set_xticklabels([config.LM_TO_TECH_REPORT_NOTATION[y + 1] for y in range(0, 32)])
        ax.set_xlim(0, 33)
        ax.set_yticks([2 * x for x in range(0, 16)])
        plt.xticks(rotation=45, ha='right')

        ax.set_ylabel('Distance from GT (mm)', fontsize=20)
        ax.set_xlabel('Landmark notation', fontsize=20)

        ax.get_yaxis().tick_left()
        ax.get_xaxis().tick_bottom()

        ax.legend()

        name = 'box-plots-lds-NestedUNet-' + consensus + '-100-views.pgf'
        plt.ylim(top=30)
        plt.tight_layout()
        plt.savefig(name)

    def get_matching_acceptances(self, consensus='RANSAC'):
        """Method plots matching acceptances graphs of individual landmarks."""

        assert consensus in ['RANSAC', 'CENTROID']

        acceptance_values = [_ for _ in range(0, 40, 2)]

        plt.figure(figsize=(7, 11))
        plt.axvline(2, linestyle='--', color='r', label='Acceptable distance (2 mm)')

        if consensus == 'RANSAC':
            mask = ~np.isnan(self.distances_RANSAC)
            filtered_data = [d[m] for d, m in zip(self.distances_RANSAC.T, mask.T)]
        else:
            mask = ~np.isnan(self.distances_CENTROID)
            filtered_data = [d[m] for d, m in zip(self.distances_CENTROID.T, mask.T)]

        for idx, landmark in enumerate(filtered_data):
            acceptances = []
            for acceptance_value in acceptance_values:
                if len(landmark) != 0:
                    acceptances.append(len([1 for i in landmark if i <= acceptance_value]) / len(landmark) * 100)

            if len(landmark) != 0:
                plt.plot(acceptance_values, acceptances,
                         linestyle=self.get_random_line_type(),
                         color=self.get_random_color(),
                         marker=self.get_random_marker(),
                         label=config.LM_TO_TECH_REPORT_NOTATION[idx + 1])

        plt.xlabel('Acceptance value (mm)', fontsize=20)
        plt.ylabel('Acceptance percentage', fontsize=20)
        ax = plt.gca()
        ax.grid(True)

        plt.legend(fontsize=10)
        plt.tight_layout()

        name = 'matching-acceptances-NestedUNet-' + consensus + '-100-views.pgf'
        plt.savefig(name)

    @staticmethod
    def get_random_color():
        clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        return random.choice(clrs)

    @staticmethod
    def get_random_line_type():
        types = ['-', '--', '-.', ':']

        return random.choice(types)

    @staticmethod
    def get_random_marker():
        markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'x', 'D']

        return random.choice(markers)
