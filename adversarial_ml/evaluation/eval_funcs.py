import os
import numpy as np

from adversarial_ml.utils.paths import misc_path
from adversarial_ml.evaluation.evaluation_utils import get_network, get_data


def compute_clean_accuracy(log_file_path):
    """ Computes accuracy on clean data based on log-file. """
    with open(log_file_path, 'r') as fp:
        log = [line.rstrip() for line in fp]

    indices = np.array(log[0].split(','))
    log = [line.split(',') for line in log[1:]]
    true_idx = np.argmax(indices == 'true')
    orig_idx = np.argmax(indices == 'orig')

    cols = np.array(log)[:, [true_idx, orig_idx]]
    corr = np.sum(cols[:, 0] == cols[:, 1])
    return corr / len(log)


def compute_avg_std_iterations(log_file_path):
    """ Computes average/std of number of iterations for an adversary. """
    with open(log_file_path, 'r') as fp:
        log = [line.rstrip() for line in fp]

    indices = np.array(log[0].split(','))
    log = np.array([line.split(',') for line in log[1:]])
    epoch_idx = np.argmax(indices == 'epoch')
    conv_idx = np.argmax(indices == 'converged')

    iterations = [int(log[i, epoch_idx]) for i in range(len(log)) if log[i, conv_idx] == 'True']
    return np.mean(iterations), np.std(iterations), len(iterations)


def compute_quart_iterations(log_file_path):
    """ Computes and returns 0.25, 0.5 (median) and 0.75 quantile of iterations. """
    with open(log_file_path, 'r') as fp:
        log = [line.rstrip() for line in fp]

    indices = np.array(log[0].split(','))
    log = np.array([line.split(',') for line in log[1:]])
    epoch_idx = np.argmax(indices == 'epoch')
    conv_idx = np.argmax(indices == 'converged')

    iterations = [int(log[i, epoch_idx]) for i in range(len(log)) if log[i, conv_idx] == 'True']
    return np.quantile(iterations, 0.25), np.quantile(iterations, 0.5), np.quantile(iterations, 0.5)


def compute_avg_std_snr(log_file_path):
    """ Computes average/std of SNR for an adversary. """
    with open(log_file_path, 'r') as fp:
        log = [line.rstrip() for line in fp]

    indices = np.array(log[0].split(','))
    log = np.array([line.split(',') for line in log[1:]])
    db_idx = np.argmax(indices == 'db')
    conv_idx = np.argmax(indices == 'converged')

    snr = [float(log[i, db_idx]) for i in range(len(log)) if log[i, conv_idx] == 'True']
    return np.mean(snr), np.std(snr), len(snr)


def compute_adversarial_accuracy(log_file_path):
    """ Computes adversarial accuracy based on log-file. """
    with open(log_file_path, 'r') as fp:
        log = [line.rstrip() for line in fp]

    indices = np.array(log[0].split(','))
    log = np.array([line.split(',') for line in log[1:]])
    true_idx = np.argmax(indices == 'true')
    new_idx = np.argmax(indices == 'new')

    cols = np.array(log)[:, [true_idx, new_idx]]
    corr = np.sum(cols[:, 0] == cols[:, 1])
    return corr / len(log)


def compute_avg_std_confidence(model_name, adversary, valid_set=True):
    """ Computes average/std of confidence in predictions given a network only. """
    net = get_network(model_name)
    data_loader = get_data(model_name, adversary, valid_set, False)

    confs = []
    for x, y in data_loader:
        conf = net.probabilities(x).flatten()
        pred = net.predict(x)
        confs.append(conf[pred.item()])
    confs = np.array(confs)

    return np.mean(confs), np.std(confs), len(confs)


if __name__ == '__main__':
    log_path = os.path.join(misc_path, 'logs/test_torch16s1f.csv')
    print(compute_clean_accuracy(log_path))
    print(compute_avg_std_iterations(log_path))
    print(compute_quart_iterations(log_path))
    print(compute_avg_std_snr(log_path))
    print(compute_adversarial_accuracy(log_path))
    print(compute_avg_std_confidence('torch16s1f', 'test'))
