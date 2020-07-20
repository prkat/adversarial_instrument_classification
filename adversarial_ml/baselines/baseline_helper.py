import os

from shutil import copyfile
from adversarial_ml.utils.paths import misc_path
from adversarial_ml.utils.logging_utils import Logger
from adversarial_ml.utils.attack_utils import check_and_create_adv_dirs


def prep_logger(param_file_path, params):
    """ Copies configuration file into logging directory and prepares logger. """
    ad_log_path = os.path.join(misc_path, 'logs')
    copyfile(param_file_path, os.path.join(ad_log_path, '{}_{}.txt'.format(params.experiment, params.model_to_attack)))
    log_file_path = os.path.join(ad_log_path, '{}_{}.csv'.format(params.experiment, params.model_to_attack))
    logger = Logger(log_file_path, columns=['file', 'epoch', 'db', 'converged', 'true', 'orig', 'new'])
    return logger


def prep_dir(experiment):
    """ Prepares directory for storing adversarial examples. """
    check_and_create_adv_dirs()
    s_path = os.path.join(misc_path, 'adversaries/{}/'.format(experiment))
    if not os.path.exists(s_path):
        os.mkdir(s_path)
    return s_path
