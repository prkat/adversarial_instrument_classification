import os
import torch
import torchaudio
import numpy as np


def get_torch_spec(file, sample_path=None, to_db=True, norm_file_path=None, dtype=torch.Tensor):
    """
    Computes Mel-spectrograms with torchaudio.
    :param file: Name of file containing audio or array containing audio directly.
    :param sample_path: Path of file, if sample still needs to be read.
    :param to_db: Whether to convert to dB.
    :param norm_file_path: File pointing to norm_file, if normalisation is required.
    :param dtype: Return type.
    :return: Mel-spectrogram
    """
    raw_data = _get_torch_sample(file, sample_path)
    # compute mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512,
                                                         n_mels=100, f_min=40.).to(raw_data.device)
    mel_spec = mel_transform.forward(raw_data)

    if to_db:
        db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.)
        mel_spec = db_transform.forward(mel_spec)
    if norm_file_path is not None:
        mel_spec = normalise(mel_spec, norm_file_path)

    if dtype != torch.Tensor and dtype != np.ndarray:
        raise ValueError('Please choose between pytorch tensor and numpy array!')
    return mel_spec.view(1, 100, -1) if dtype == torch.Tensor else mel_spec.numpy()


def _get_torch_sample(file, sample_path):
    """ Prepares torch audio sample, reads it first if necessary. """
    if isinstance(file, str):
        if sample_path is None:
            raise ValueError('Define path for this sample first!')
        raw_data, _ = torchaudio.load(os.path.join(sample_path, file))
    else:
        raw_data = torch.tensor(file) if isinstance(file, np.ndarray) else file

    raw_data = raw_data.float().view(1, -1)
    return raw_data


def normalise(feature, norm_file_path):
    """ Normalises a given feature (e.g. spectrogram, mfcc) with statistics given at norm_file_path. """
    std = np.loadtxt(norm_file_path.format('std'), delimiter=',')
    mean = np.loadtxt(norm_file_path.format('mean'), delimiter=',')

    if isinstance(feature, np.ndarray):
        feature = ((feature.transpose() - mean) / std).transpose()
    else:
        orig_shape = feature.shape
        mean = torch.tensor(mean).float().to(feature.device)
        std = torch.tensor(std).float().to(feature.device)
        feature = torch.transpose(((torch.transpose(feature.squeeze(), 0, 1) - mean) / std), 0, 1)
        feature = feature.view(orig_shape)
    return feature
