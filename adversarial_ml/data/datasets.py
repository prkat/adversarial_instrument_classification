import os
import torch
import torchaudio
import numpy as np

from torch.utils.data import Dataset
from adversarial_ml.utils.paths import csv_path
from adversarial_ml.utils.utils import read_file
from adversarial_ml.data.feature_extraction import get_torch_spec, normalise
from adversarial_ml.data.data_utils import get_train_label_dict, get_test_label_dict, \
    get_unique_label_set, read_feature, divide_in_frames


class RawDataset(Dataset):
    """ Dataset for single-label musical audio-files, returns raw waveforms. """
    def __init__(self, filename, data_path):
        self.filenames = filename if isinstance(filename, list) else read_file(csv_path, filename)
        self.data_path = data_path
        self.labels = get_train_label_dict()
        if not set(self.filenames).issubset(set(self.labels.keys())):
            self.labels.update(get_test_label_dict())
        self.label_map = {l: i for i, l in enumerate(sorted(get_unique_label_set()))}
        self.filenames = sorted(list(set(self.filenames).intersection(set(self.labels.keys()))))

    def __len__(self):
        return len(self.filenames)

    def _read_data(self, clip):
        if os.path.exists(os.path.join(self.data_path, 'train_curated')):
            data, _ = torchaudio.load(os.path.join(self.data_path, 'train_curated', clip))
        else:
            data, _ = torchaudio.load(os.path.join(self.data_path, clip))
        data = data.view(1, -1).float()
        return data

    def __getitem__(self, index):
        clip = self.filenames[index]
        data = self._read_data(clip)

        # get label
        [tag] = self.labels.get(clip)
        label = torch.tensor(self.label_map[tag]).long()
        return data, label


class AudioDataset(Dataset):
    """ Dataset for single-label musical audio-files, returns (pre-computed) Mel-spectrograms or MFCCs. """
    def __init__(self, filename, data_path, feature_dict, norm_file_path=None, d_path_backup=None):
        self.filenames = filename if isinstance(filename, list) else read_file(csv_path, filename)
        self.data_path, self.back_up_data_path = data_path, d_path_backup
        self.feature_dict = feature_dict
        self.norm_file_path = norm_file_path

        if feature_dict.feature != 'torch':
            raise NotImplementedError('Please define valid feature! (`torch`)')

        if feature_dict.pre_computed:
            self.get_features = read_feature
        else:
            self.get_features = get_torch_spec

        self.labels = get_train_label_dict()
        if not set(self.filenames).issubset(set(self.labels.keys())):
            self.labels.update(get_test_label_dict())
        self.label_map = {l: i for i, l in enumerate(sorted(get_unique_label_set()))}
        self.filenames = sorted(list(set(self.filenames).intersection(set(self.labels.keys()))))

    def __len__(self):
        return len(self.filenames)

    def _read_data(self, path, clip):
        if os.path.exists(os.path.join(path, 'train_curated')):
            path = os.path.join(path, 'train_curated')
        data = self.get_features(file=clip, sample_path=path)
        # normalisation
        data = self._normalise(data)

        # if feature length given, divide in windows and pick one randomly
        # otherwise, make sure feature is long enough for network
        windows = divide_in_frames(data, self.feature_dict.feature_length)
        [win_idx] = np.random.choice(windows.shape[0], 1)
        data = windows[win_idx]
        return data.view(1, 100, -1)

    def _normalise(self, data):
        if self.norm_file_path is None and not self.feature_dict.sample_wise_norm:
            return data
        elif self.feature_dict.sample_wise_norm:
            e = 1e-5
            mean = torch.mean(data.squeeze(), dim=-1)
            std = torch.std(data.squeeze(), dim=-1)
            return torch.transpose(((torch.transpose(data.squeeze(), 0, 1) - mean) / (std + e)), 0, 1).view(1, 100, -1)
        else:
            # normalise with pre-computed mean/std
            return normalise(data, self.norm_file_path).view(1, 100, -1)

    def __getitem__(self, index):
        clip = self.filenames[index]
        try:
            data = self._read_data(self.data_path, clip)
        except OSError:
            data = self._read_data(self.back_up_data_path, clip)

        [tag] = self.labels.get(clip)
        label = torch.tensor(self.label_map[tag]).long()
        return data, label
