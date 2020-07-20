import os
import torch
import numpy as np

from shutil import copyfile
from adversarial_ml.utils.utils import load_net
from adversarial_ml.utils.logging_utils import Logger
from adversarial_ml.utils.paths import d_path, log_path, misc_path
from adversarial_ml.data.data_utils import get_unique_label_set
from adversarial_ml.data.data_helper import make_get_feature, get_raw_test_loader
from adversarial_ml.utils.attack_utils import get_feature_fun, get_files, get_net, check_and_create_adv_dirs


def get_random_target(orig_prediction, rand_state):
    """ Returns target class that is not equal to original prediction. """
    nr_classes = len(get_unique_label_set())
    pos_classes = [cl for cl in np.arange(nr_classes) if cl != orig_prediction]
    target = rand_state.choice(pos_classes, 1)
    return torch.tensor(target)


def get_label_from_idx(idx):
    """ Given label index, returns string representation. """
    unique_labels_dict = {l: k for l, k in enumerate(sorted(list(get_unique_label_set())))}
    unique_labels_dict.update({-1: 'random'})
    return unique_labels_dict[idx.item()].lower()


def check_target(target):
    """ Checks validity of defined target. """
    if target.lower() == 'random':
        return torch.tensor([-1])
    unique_labels = sorted(list(get_unique_label_set()))
    unique_labels_dict = {k: l for l, k in enumerate(unique_labels)}

    if isinstance(target, int) and -1 >= target > len(unique_labels):
        return torch.tensor([target])

    for label in unique_labels_dict.keys():
        if label.lower() == target.lower():
            return torch.tensor([unique_labels_dict[label]])

    raise ValueError('Please define valid target!')


def prep_net(model_name):
    """ Preps and returns network. """
    net = load_net(model_name, get_net())
    net.eval()
    return net


def prep_logger(param_file_path, params, target):
    """ Copies configuration file into logging directory and prepares logger. """
    target = target.lower()
    ad_log_path = os.path.join(misc_path, 'logs')
    copyfile(param_file_path, os.path.join(ad_log_path, '{}_{}_{}.txt'.format(params.experiment,
                                                                              params.model_to_attack, target)))
    log_file_path = os.path.join(ad_log_path, '{}_{}_{}.csv'.format(params.experiment, params.model_to_attack, target))
    logger = Logger(log_file_path, columns=['file', 'epoch', 'db', 'converged', 'true', 'orig', 'new', 'target'])
    return logger


def prep_dir(experiment, target):
    """ Prepares directory for storing adversarial examples. """
    check_and_create_adv_dirs()
    s_path = os.path.join(misc_path, 'adversaries/{}_{}/'.format(experiment, target.lower()))
    if not os.path.exists(s_path):
        os.mkdir(s_path)
    return s_path


def prep_feature_data(params):
    """ Prepares data-loader and feature computation function. """
    model_name = params.model_to_attack
    feature_fun = get_feature_fun(model_name)
    files = sorted(get_files(params.valid))
    data_loader = get_raw_test_loader(files, d_path if params.valid else os.path.join(d_path, 'test'))
    get_feature = make_get_feature(feature_fun, os.path.join(log_path, model_name, '{}.csv'))
    return get_feature, data_loader
