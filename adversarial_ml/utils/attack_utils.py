import os
import numpy as np

from adversarial_ml.utils.paths import d_path, adversary_path, misc_path
from adversarial_ml.utils.avgpool_cnn import AveragePoolCNN
from adversarial_ml.data.feature_extraction import get_torch_spec
from adversarial_ml.data.data_helper import get_single_label_files


def get_feature_fun(model_name):
    """ Given name of saved model, returns according norm-file-name for features. """
    if 'torch' in model_name:
        return get_torch_spec
    else:
        raise NotImplementedError('Invalid feature')


def get_net():
    """ Given model name, returns initialised model architecture. """
    return AveragePoolCNN(1, 12)


def get_files(valid=True):
    """ Returns test or validation files. """
    if valid:
        tot_files = sorted(get_single_label_files())
        rng = np.random.RandomState(21)
        rng.shuffle(tot_files)
        split_idx = int(len(tot_files) * 0.75)
        return tot_files[split_idx:]
    else:
        return os.listdir(os.path.join(d_path, 'test'))


def snr(x, x_hat):
    """ SNR computation according to https://github.com/coreyker/dnn-mgr/blob/master/utils/comp_ave_snr.py. """
    ign = 2048
    lng = min(x.shape[-1], x_hat.shape[-1])
    ratio = 20 * np.log10(np.linalg.norm(x[..., ign:lng - ign - 1]) /
                          np.linalg.norm(np.abs(x[..., ign:lng - ign - 1] - x_hat[..., ign:lng - ign - 1]) + 1e-12))
    return ratio


def check_and_create_adv_dirs():
    """ Checks existence of adversary/logs directory in misc folder - creates directories if necessary. """
    if not os.path.exists(adversary_path):
        os.mkdir(adversary_path)
    if not os.path.exists(os.path.join(misc_path, 'logs')):
        os.mkdir(os.path.join(misc_path, 'logs'))
