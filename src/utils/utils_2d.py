"""
    :filename utils_2d.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Utilities needed for the parts of the pipeline in 2D.
"""

import numpy as np

from scipy.ndimage.filters import maximum_filter

import config


def non_maxima_suppression(img, perf_meas, ld_num, heatmaps_max_vals=None):
    """
    Function finds a peak value of a heatmap by the NMS algorithm.

    :param heatmaps_max_vals:
    :param img: Input heatmap.
    :param perf_meas: Instance of the PerformanceMeasure class.
    :return: Display coordinates of the prediction and a flag defining the landmark presence prediction in the heatmap.
    """
    try:
        preds_np = img.detach().numpy()
    except TypeError:
        preds_np = img.cpu().detach().numpy().astype('float32')

    if heatmaps_max_vals is not None:
        if ld_num not in heatmaps_max_vals:
            heatmaps_max_vals[ld_num] = list()
        heatmaps_max_vals[ld_num].append(preds_np.max())

    '''
    Detection of the maximal value after the threshold is applied.
    If all values are less than given threshold, the max value will be 0.
    Otherwise, the maximal value is assigned to the variable.
    '''
    max_val_after_threshold = np.amax(np.where(preds_np < config.BIN_CLASSIFIER_THRESHOLD, 0, preds_np))

    if max_val_after_threshold == 0:
        is_present = False

        if perf_meas is not None:
            perf_meas.heatmap_certainties_vals_missing.append(np.amax(preds_np))
    else:
        is_present = True
        if perf_meas is not None:
            perf_meas.heatmap_certainties_vals_present.append(np.amax(preds_np))

    '''NMS algorithm.'''
    preds_np = np.where(preds_np < 0, 0, preds_np)
    after_nms = preds_np * (preds_np == maximum_filter(preds_np, footprint=np.ones((128, 128))))
    x, y = np.unravel_index(after_nms.argmax(), after_nms.shape)

    return x, y, is_present
