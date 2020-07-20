import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists

from adversarial_ml.utils.paths import log_path, d_path
from adversarial_ml.baselines.baseline_helper import prep_dir, prep_logger
from adversarial_ml.utils.utils import get_params, load_net, save_adv_example
from adversarial_ml.data.data_helper import get_raw_test_loader, make_get_feature
from adversarial_ml.utils.attack_utils import get_feature_fun, get_net, get_files, snr


def _pgdn(net, params, x, y, get_feature, dist):
    """ Projected Gradient Descent on negative loss function. """
    # see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    eps = params.epsilon
    orig_pred = net.predict(get_feature(x)).item()

    delta = torch.zeros(x.shape).to(x.device)
    if params.rand_start:
        delta = dist.sample(x.shape)
    delta.requires_grad = True

    optimizer = optim.Adam([delta], lr=params.lr)
    loss_func = nn.CrossEntropyLoss()

    for i in range(1, params.max_iter + 1):
        new_x = x + torch.clamp(delta, -eps, eps).to(x.device)
        new_x = torch.clamp(new_x, min=-1., max=1.).to(x.device)
        logits = net(get_feature(new_x))
        loss = loss_func(logits, y)

        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            delta.grad = torch.sign(delta.grad) * -1.
        optimizer.step()

        cur_pred = net.predict(get_feature(new_x)).item()
        db = snr(x.cpu().numpy(), new_x.detach().cpu().numpy())
        if cur_pred != orig_pred:
            return new_x.view(1, -1), db, i

        diff_to_next = torch.abs(logits[:, orig_pred] - sorted(logits.squeeze())[-2]).item()
        msg = '\rep {}/{}; diff to next: {}; current db: {}'
        print(msg.format(i, params.max_iter, diff_to_next, db), flush=True, end='')

    return None, None, -1


def run_pgdn(net, data, get_feature, logger, ad_save_path, params):
    """ Computes PGD on negative loss for given data. """
    dist = dists.Uniform(-params.epsilon, params.epsilon)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    for i, (x, y) in enumerate(data):
        print('pgd sample {}/{}'.format(i + 1, len(data)))

        file = data.dataset.filenames[i]
        file = file if '.wav' in file else file + '.wav'

        ad_ex, db, converged = _pgdn(net, params, x.to(device), y.to(device), get_feature, dist)
        orig_pred = net.predict(get_feature(x.to(device))).item()
        true_pred = y.item()
        if converged >= 0:
            print('\nsave current file: {} with distortion: {}db (r:{}/o:{})'.format(file, db, true_pred, orig_pred))
            new_pred = net.predict(get_feature(ad_ex)).item()
            save_adv_example(ad_ex.cpu(), os.path.join(ad_save_path, file))
            logger.append([file, converged, db, True, true_pred, orig_pred, new_pred])
        else:
            print('\ncould not find robust adv. example for file {}'.format(file))
            logger.append([file, converged, 0, False, true_pred, orig_pred, orig_pred])


def main():
    # get params
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    param_file = os.path.join(cur_dir, 'pgdn_config.txt')
    params = get_params(param_file)

    # load model to be attacked
    model_name = params.model_to_attack
    feature_fun = get_feature_fun(model_name)
    net = load_net(model_name, get_net())

    # prep directories for experiment
    ad_save_path = prep_dir(params.experiment)
    # prep data and feature computation
    files = sorted(get_files(params.valid))
    data_loader = get_raw_test_loader(files, d_path if params.valid else os.path.join(d_path, 'test'))
    get_feature = make_get_feature(feature_fun, os.path.join(log_path, model_name, '{}.csv'))
    # prep logger
    logger = prep_logger(param_file, params)

    # run fgsm
    run_pgdn(net, data_loader, get_feature, logger, ad_save_path, params)


if __name__ == '__main__':
    main()
