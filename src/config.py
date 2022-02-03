"""
    :filename config.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Config file for:
        (i)     Network training hyperparameters,
        (ii)    Gaussian parameters,
        (iii)   input and output dimensions,
        (iv)    RANSAC parameters,
        (v)     different landmark notations.
"""

'''Hyperparameter configuration.'''
HYPERPARAMETERS = {
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,

    'lr_scheduler_patience': 5,
    'lr_scheduler_min_lr': 1e-6,
    'lr_scheduler_factor': 0.5,

    'early_stopping_patience': 20,

    'heatmap-sigma': 10,
    'heatmap-amplitude': 1,

    'ransac_iterations': 50,
    'ransac_threshold_mm': 5,

    'multi-view-N': 100
}

'''Input and output dimension configuration.'''
DIMENSIONS = {
    'original': 512,
    'input_net': 128,
    'output_net': 128
}

'''
List of all notations for all teeth landmarks. Last digit always
describes whether it is a distal(0) or mesial(1) landmark of given
tooth.
Example: Notation 141 stands for mesial landmark of tooth 14.
'''
VALID_NOTATIONS = [10, 11, 20, 21, 30, 31, 40, 41,
                   50, 51, 60, 61, 70, 71, 80, 81,
                   90, 91, 100, 101, 110, 111, 120, 121,
                   130, 131, 140, 141, 150, 151, 160, 161]

LANDMARKS_NUM = 32
MULTI_VIEW_NUM = 100

BIN_CLASSIFIER_THRESHOLD = 0.375
PARTIAL_PREDS_THRESHOLD = 15  # number of partial predictions, which uncertainty must be higher than BIN_CL_THR

'''Full teeth notation.'''
TEETH_NOTATIONS = {
    1: 'Right Third Molar',
    2: 'Right Second Molar',
    3: 'Right First Molar',
    4: 'Right Second Premolar',
    5: 'Right First Premolar',
    6: 'Right Canine',
    7: 'Right Lateral Incisor',
    8: 'Right Central Incisor',
    9: 'Left Central Incisor',
    10: 'Left Lateral Incisor',
    11: 'Left Canine',
    12: 'Left First Premolar',
    13: 'Left Second Premolar',
    14: 'Left First Molar',
    15: 'Left Second Molar',
    16: 'Left Third Molar',
    17: 'Left Third Molar',
    18: 'Left Second Molar',
    19: 'Left First Molar',
    20: 'Left Second Premolar',
    21: 'Left First Premolar',
    22: 'Left Canine',
    23: 'Left Lateral Incisor',
    24: 'Left Central Incisor',
    25: 'Right Central Incisor',
    26: 'Right Lateral Incisor',
    27: 'Right Canine',
    28: 'Right First Premolar',
    29: 'Right Second Premolar',
    30: 'Right First Molar',
    31: 'Right Second Molar',
    32: 'Right Third Molar'
}

SCENE_NAMES = {
    1: '8D',
    2: '8M',
    3: '7D',
    4: '7M',
    5: '6D',
    6: '6M',
    7: '5D',
    8: '5M',
    9: '4D',
    10: '4M',
    11: '3D',
    12: '3M',
    13: '2D',
    14: '2M',
    15: '1D',
    16: '1M',
    17: '1M',
    18: '1D',
    19: '2M',
    20: '2D',
    21: '3M',
    22: '3D',
    23: '4M',
    24: '4D',
    25: '5M',
    26: '5D',
    27: '6M',
    28: '6D',
    29: '7M',
    30: '7D',
    31: '8M',
    32: '8D'
}

SCENE_NAMES_LONG = {
    1: 'Third Molar (Distal)',
    2: 'Third Molar (Mesial)',
    3: 'Second Molar (Distal)',
    4: 'Second Molar (Mesial)',
    5: 'First Molar (Distal)',
    6: 'First Molar (Mesial)',
    7: 'Second Premolar (Distal)',
    8: 'Second Premolar (Mesial)',
    9: 'First Premolar (Distal)',
    10: 'First Premolar (Mesial)',
    11: 'Canine (Distal)',
    12: 'Canine (Mesial)',
    13: 'Lateral Incisor (Distal)',
    14: 'Lateral Incisor (Mesial)',
    15: 'Central Incisor (Distal)',
    16: 'Central Incisor (Mesial)',
    17: 'Central Incisor (Mesial)',
    18: 'Central Incisor (Distal)',
    19: 'Lateral Incisor (Mesial)',
    20: 'Lateral Incisor (Distal)',
    21: 'Canine (Mesial)',
    22: 'Canine (Distal)',
    23: 'First Premolar (Mesial)',
    24: 'First Premolar (Distal)',
    25: 'Second Premolar (Mesial)',
    26: 'Second Premolar (Distal)',
    27: 'First Molar (Mesial)',
    28: 'First Molar (Distal)',
    29: 'Second Molar (Mesial)',
    30: 'Second Molar (Distal)',
    31: 'Third Molar (Mesial)',
    32: 'Third Molar (Distal)'
}

'''Short landmarks notation.'''
LM_TO_TECH_REPORT_NOTATION = {
    1: 'L8D',
    2: 'L8M',
    3: 'L7D',
    4: 'L7M',
    5: 'L6D',
    6: 'L6M',
    7: 'L5D',
    8: 'L5M',
    9: 'L4D',
    10: 'L4M',
    11: 'L3D',
    12: 'L3M',
    13: 'L2D',
    14: 'L2M',
    15: 'L1D',
    16: 'L1M',
    17: 'R1M',
    18: 'R1D',
    19: 'R2M',
    20: 'R2D',
    21: 'R3M',
    22: 'R3D',
    23: 'R4M',
    24: 'R4D',
    25: 'R5M',
    26: 'R5D',
    27: 'R6M',
    28: 'R6D',
    29: 'R7M',
    30: 'R7D',
    31: 'R8M',
    32: 'R8D'
}

'''CSV landmarks notation.'''
TOOTH_TO_CSV_NOTATION = {
    1: 10,
    2: 11,
    3: 20,
    4: 21,
    5: 30,
    6: 31,
    7: 40,
    8: 41,
    9: 50,
    10: 51,
    11: 60,
    12: 61,
    13: 70,
    14: 71,
    15: 80,
    16: 81,
    17: 90,
    18: 91,
    19: 100,
    20: 101,
    21: 110,
    22: 111,
    23: 120,
    24: 121,
    25: 130,
    26: 131,
    27: 140,
    28: 141,
    29: 150,
    30: 151,
    31: 160,
    32: 161
}

TOTAL_TRAIN_SET_COUNT = 337

TRAIN_SET_LM_COUNT = {
    1: 40,
    2: 40,
    3: 232,
    4: 232,
    5: 248,
    6: 248,
    7: 285,
    8: 285,
    9: 303,
    10: 303,
    11: 327,
    12: 327,
    13: 322,
    14: 322,
    15: 312,
    16: 312,
    17: 314,
    18: 314,
    19: 313,
    20: 313,
    21: 325,
    22: 325,
    23: 301,
    24: 301,
    25: 279,
    26: 279,
    27: 255,
    28: 255,
    29: 242,
    30: 242,
    31: 43,
    32: 43
}

CLASS_BALANCE_BETA = 0.99999
DIFFICULTY_BALANCE_THETA = 4
