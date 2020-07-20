import torch
import numpy as np

from torch.utils.data import DataLoader
from adversarial_ml.data.datasets import RawDataset
from adversarial_ml.data.data_utils import circular_pad, get_train_label_dict
from adversarial_ml.data.feature_extraction import normalise, get_torch_spec


def compute_std_mean(files, sample_path):
    """ Returns mean and standard deviation of features for given files. """
    fun = get_torch_spec

    x = np.zeros((1, 100))
    x2 = np.zeros((1, 100))
    n = 0

    for file in files:
        feat = fun(file, sample_path, norm_file_path=None, to_db=True, dtype=np.ndarray)
        n += feat.shape[-1]
        x += np.sum(feat, axis=-1)
        x2 += np.sum(feat ** 2, axis=-1)

    mean = x / n
    std = np.sqrt((x2 / n) - mean ** 2)
    return mean, std


def get_single_label_files():
    """ Returns all single label training files. """
    return list(get_train_label_dict().keys())


def get_raw_test_loader(files, d_path):
    """ Returns un-shuffled DataLoader with test-files of batch-size 1. """
    ds = RawDataset(files, d_path)
    return DataLoader(ds, batch_size=1, shuffle=False)


def get_feature_norm_pad(feature_fun, file, path, norm_file_path, sample_wise_norm):
    """ Given a feature function, computes normalised and padded feature. """
    # compute feature
    feature = feature_fun(file, sample_path=path)
    # normalise if required
    if sample_wise_norm:
        mean = torch.mean(feature.squeeze(), dim=-1)
        std = torch.std(feature.squeeze(), dim=-1)
        feature = torch.transpose(((torch.transpose(feature.squeeze(), 0, 1) - mean) /
                                   (std + 1e-5)), 0, 1).view(1, 100, -1)
    elif norm_file_path is not None:
        feature = normalise(feature, norm_file_path=norm_file_path).view(1, 100, -1)
    # pad if shorter than 35
    pad_len = 35 - feature.shape[-1]
    if pad_len > 0:
        return circular_pad(feature, pad_len).view(1, 1, 100, -1)
    return feature.view(1, 1, 100, -1)


def make_get_feature(feature_fun, norm_file_path):
    """ Returns wrapper function for feature computation. """
    def get_features(file):
        return get_feature_norm_pad(feature_fun, file.view(1, -1), None, norm_file_path, False)
    return get_features
