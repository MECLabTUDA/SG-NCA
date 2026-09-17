#https://github.com/MECLabTUDA/SDS_Playground/blob/main/sds_playground/datasets/cholecseg8k/cholecseg8k_visualisation.py
import numpy as np
import torch


CLASSES = {
    -3: {
        "name": "Ignore",
        "color": [255, 50, 50],
        "rgb-hex": [0, 0, 0],
        "super-category": "Misc"
    },
    -2: {
        "name": "Ignore",
        "color": [127, 127, 127],
        "rgb-hex": [0, 0, 0],
        "super-category": "Misc"
    },
    -1: {
        "name": "Ignore",
        "color": [127, 127, 127],
        "rgb-hex": [255, 255, 255],
        "super-category": "Misc"
    },
    0: {
        "name": "Black Background",
        "color": [127, 127, 127],
        "rgb-hex": [50, 50, 50],
        "super-category": "Misc"
    },
    1: {
        "name": "Abdominal Wall",
        "color": [210, 140, 140],
        "rgb-hex": [11, 11, 11],
        "super-category": "Organ"
    },
    2: {
        "name": "Liver",
        "color": [255, 114, 114],
        "rgb-hex": [21, 21, 21],
        "super-category": "Organ"
    },
    3: {
        "name": "Gastrointestinal Tract",
        "color": [231, 70, 156],
        "rgb-hex": [13, 13, 13],
        "super-category": "Organ"
    },
    4: {
        "name": "Fat",
        "color": [186, 183, 75],
        "rgb-hex": [12, 12, 12],
        "super-category": "Organ"
    },
    5: {
        "name": "Grasper",
        "color": [170, 255, 0],
        "rgb-hex": [31, 31, 31],
        "super-category": "Instrument"
    },
    6: {
        "name": "Connective Tissue",
        "color": [255, 85, 0],
        "rgb-hex": [23, 23, 23],
        "super-category": "Organ"
    },
    7: {
        "name": "Blood",
        "color": [255, 0, 0],
        "rgb-hex": [24, 24, 24],
        "super-category": "Fluid"
    },
    8: {
        "name": "Cystic Duct",
        "color": [255, 255, 0],
        "rgb-hex": [25, 25, 25],
        "super-category": "Organ"
    },
    9: {
        "name": "L-hook Electrocautery",
        "color": [169, 255, 184],
        "rgb-hex": [32, 32, 32],
        "super-category": "Instrument"
    },
    10: {
        "name": "Gallblader",
        "color": [255, 160, 165],
        "rgb-hex": [22, 22, 22],
        "super-category": "Organ"
    },
    11: {
        "name": "Hepatic Vein",
        "color": [0, 50, 128],
        "rgb-hex": [33, 33, 33],
        "super-category": "Vein"
    },
    12: {
        "name": "Liver Ligament",
        "color": [111, 74, 0],
        "rgb-hex": [5, 5, 5],
        "super-category": "Organ"
    }
}


def get_cholecseg8k_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [127, 127, 127],  # Black Background
            [210, 140, 140],  # Abdominal Wall
            [255, 114, 114],  # Liver
            [231, 70, 156],  # Gastrointestinal Tract
            [186, 183, 75],  # Fat
            [170, 255, 0],  # Grasper
            [255, 85, 0],  # Connective Tissue
            [255, 0, 0],  # Blood
            [255, 255, 0],  # Cystic Dust
            [169, 255, 184],  # L-hook Electrocautery
            [255, 160, 165],  # Gallblader
            [0, 50, 128],  # Hepatic Vein
            [111, 74, 0],  # Liver Ligament
        ]
    )


def get_cholecseg8k_float_cmap():
    return torch.from_numpy(get_cholecseg8k_colormap())/255.0