import os
import torch
import numpy as np
import torch.nn.functional as fu

from math import ceil
from adversarial_ml.utils.paths import csv_path


def get_unique_label_set():
    unique_label_set = {'Accordion', 'Acoustic_guitar', 'Bass_drum', 'Bass_guitar', 'Electric_guitar', 'Female_singing',
                        'Glockenspiel', 'Gong', 'Harmonica', 'Hi-hat', 'Male_singing', 'Marimba_and_xylophone'}
    return unique_label_set


def get_train_label_dict():
    return read_label_dict(os.path.join(csv_path, 'train_curated_post_competition.csv'))


def get_test_label_dict():
    return read_label_dict(os.path.join(csv_path, 'test_post_competition.csv'))


def read_label_dict(file_path):
    """ Returns dictionary containing ground-truth labels mapped to single-label file-names. """
    labels = get_unique_label_set()
    data = {}
    with open(file_path) as fp:
        fp.readline()
        for line in fp:
            row = line.rstrip().replace('"', '').split(',')
            intersec = list(labels.intersection(set(row[1:])))
            if len(intersec) == 1:
                data[row[0]] = intersec
    return data


def read_feature(file, sample_path):
    """ Reads pre-computed feature from a .npy file. """
    spec = np.load(os.path.join(sample_path, file.replace('.wav', '.npy')))
    return torch.tensor(spec).view(1, 100, -1)


def circular_pad(x, padding_length):
    """ Performs circular padding iteratively, until padding_length is reached. """
    pre_padding = padding_length // 2
    post_padding = padding_length - pre_padding

    while padding_length > 0:
        possible_pad = x.shape[-1]
        x = fu.pad(x, [min(pre_padding, possible_pad), min(post_padding, possible_pad)], 'circular')
        pre_padding -= min(pre_padding, possible_pad)
        post_padding -= min(post_padding, possible_pad)
        padding_length = pre_padding + post_padding

    return x


def divide_in_frames(x, width=116):
    """ Divides data and according labels into multiple half-overlapping windows. Also does padding. """
    # if width is None, return original data
    if width is None:
        if x.shape[-1] >= 35:
            return x
        else:
            return circular_pad(x, 35 - x.shape[-1])
    jump_size = width // 2
    num_windows = ceil((x.shape[-1] - width) / jump_size) + 1
    num_windows = num_windows if num_windows > 0 else 1

    pad_len = (width + (num_windows - 1) * jump_size) - x.shape[-1]
    x = circular_pad(x, pad_len)

    new_x = torch.stack([x[..., i * jump_size:i * jump_size + width] for i in range(num_windows)])
    return new_x
