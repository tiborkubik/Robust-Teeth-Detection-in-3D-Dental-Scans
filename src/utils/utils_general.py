"""
    :filename utils_general.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    This file contains utilities for general purpose, usually plots, color getters etc.
"""

import matplotlib.pyplot as plt

from skimage.transform import resize

import config


def plot_model_with_landmark_heatmap(dataset):
    """
    Function plots an example of a depth map and corresponding heatmaps of landmarks.
    It validates the correctness of dataset initialisation.

    :param dataset: Instance of the JawDataset class used for network training.
    """

    plt.imshow(dataset[0]['image'], cmap='gray', origin='lower')
    plt.imshow(dataset[0]['landmarks'], alpha=0.7)
    plt.show(block=True)


def plot_input_batch(image_batch, landmark_batch):
    """
    Function plots one training/validation batch with corresponding landmarks.

    :param image_batch: Batch of input depth maps.
    :param landmark_batch: Batch of ground truth landmark heatmaps.
    """

    fig, axes = plt.subplots(nrows=1, ncols=config.HYPERPARAMETERS['batch_size'], figsize=(100, 100), dpi=50)

    for batch_im_idx in range(0, config.HYPERPARAMETERS['batch_size']):
        axes[batch_im_idx].imshow(resize(image_batch[batch_im_idx].permute(1, 2, 0), (112, 112)), cmap='gray')
        axes[batch_im_idx].imshow(landmark_batch[batch_im_idx].permute(1, 2, 0), alpha=0.7)

    fig.savefig('example_batch.png')


def plot_loss(tr_loss, val_loss, fig_name):
    """
    Function to plot and save the loss function of the training procedure.
    Function plots both training and validation loss.

    :param tr_loss: List of training loss values throughout the epochs.
    :param val_loss: List of validation loss values throughout the epochs.
    :param fig_name: Name of the output image.
    """

    plt.figure()
    plt.plot(range(1, len(tr_loss)+1), tr_loss, label="Training loss")
    plt.plot(range(1, len(val_loss)+1), val_loss, label="Validation loss")

    min_poss = val_loss.index(min(val_loss)) + 1
    plt.axvline(min_poss, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.ylim(0, 5.0)
    plt.xlim(0, len(tr_loss) + 1)
    plt.grid(True)

    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_name)

    plt.clf()
    plt.cla()
    plt.close()


def get_landmark_name_pred(jaw_type, lm_id, pred_type):
    """
    Function returns the name for given landmark id, suited for the prediction presentation in eval.

    :param jaw_type: Type of given jaw. According to this value, the tooth number is re-calculated.
    :param lm_id: Landmark id.
    :param pred_type: The type of prediction - CENTROID or RANSAC.

    :return: String in proper format to be displayed in the scene.
    """

    assert jaw_type in ['lower', 'upper']
    assert pred_type in ['RANSAC', 'CENTROID']

    lm_id = int(lm_id)
    lm_id += 1
    if lm_id < 17 and lm_id % 2 == 1:
        mesial_distal = 'Distal'
        lm_id += 1
    elif lm_id < 17 and lm_id % 2 == 0:
        mesial_distal = 'Mesial'
    elif lm_id >= 17 and lm_id % 2 == 1:
        mesial_distal = 'Mesial'
        lm_id += 1
    else:
        mesial_distal = 'Distal'
    lm_id /= 2

    if jaw_type == 'upper':
        tooth_name = config.TEETH_NOTATIONS[lm_id]
    else:
        lm_id += 16
        tooth_name = config.TEETH_NOTATIONS[lm_id]

    return '[' + pred_type + '] ' + str(int(lm_id)) + ' ' + mesial_distal + ', ' + tooth_name


def get_landmark_name_GT(jaw_type, lm_id):
    """
    Function returns the string value of ground truth landmark, suited for the presentation in 3D scene.

    :param jaw_type: Type of given jaw. According to this value, the tooth number is re-calculated.
    :param lm_id: Landmark id.

    :return: String in proper format to be displayed in the scene.
    """

    assert jaw_type in ['lower', 'upper']

    lm_id = int(lm_id)
    mesial_distal_flag = lm_id % 10
    lm_id = lm_id // 10

    if lm_id < 9 and mesial_distal_flag == 0:
        mesial_distal = 'Distal'
    elif lm_id < 9 and mesial_distal_flag == 1:
        mesial_distal = 'Mesial'
    elif lm_id >= 9 and mesial_distal_flag == 0:
        mesial_distal = 'Mesial'
    else:
        mesial_distal = 'Distal'

    if jaw_type == 'upper':
        tooth_name = config.TEETH_NOTATIONS[lm_id]
    else:
        lm_id += 16
        tooth_name = config.TEETH_NOTATIONS[lm_id]
    return '[GT] ' + str(lm_id) + ' ' + mesial_distal + ', ' + tooth_name


def get_distance_meter_color(distance):
    """
    Function transforms the normalized distance of two points into a RGB color.
    This color is interpolated between Green (0, 255, 0) through Yellow (255, 255, 0) to Red (255, 0, 0).

    :param distance: Normalized distance between GT and prediction, in range (0, 1).

    :return: (R, G, B) representation of the distance.
    """

    R, G, B = 0.0, 0.0, 0.0

    '''From Green to Yellow for the first half.'''
    if 0 <= distance < 0.5:
        G = 1.0
        R = 2 * distance

    '''From Yellow to Red in the second half.'''
    if 0.5 <= distance <= 1:
        R = 1.0
        G = 1.0 - 2 * (distance - 0.5)

    return R, G, B
