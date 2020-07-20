import torch.optim as optim

from adversarial_ml.utils.attack_utils import snr
from adversarial_ml.attacks.attack_helper import *
from adversarial_ml.utils.utils import get_params, save_adv_example
from adversarial_ml.attacks.target_losses import cw_loss, multi_scale_cw_loss


def update(x, delta, target, net, get_feature, optimiser, loss_fun, params):
    """ Updates perturbation delta in-place. """
    logits = net(get_feature(x + delta))

    optimiser.zero_grad()
    loss = loss_fun(logits, target, x, delta, params.alpha)
    loss.backward()
    if params.sign:
        with torch.no_grad():
            delta.grad /= torch.abs(delta.grad)
            delta.grad[delta.grad != delta.grad] = 0.  # get rid of NANs
    optimiser.step()

    with torch.no_grad():
        if params.clipping:
            delta.clamp_(min=-params.clip_eps, max=params.clip_eps)
        if params.rescale:
            delta.mul_(params.rescale_factor)


def target_attack(x, target, net, get_feature, params):
    delta = torch.zeros(x.shape).to(x.device)
    delta.requires_grad = True
    optimiser = optim.Adam([delta], lr=params.lr)
    loss_fun = cw_loss if params.attack.lower() == 'cw' else multi_scale_cw_loss

    for i in range(1, params.max_iterations + 1):
        update(x, delta, target, net, get_feature, optimiser, loss_fun, params)

        # get current logits/prediction, snr...
        logits = net(get_feature(x + delta))
        cur_snr = snr(x.cpu().numpy(), (x + delta).cpu().detach().numpy())
        cur_pred = torch.argmax(logits).item()

        if cur_pred == target.item():
            # found adversarial example, return
            return (x + delta).view(1, -1).detach(), cur_snr, i

        cur_diff = logits[0, cur_pred].item() - logits[0, target.item()].item()
        print('\rep {}/{}; current diff: {}; current snr: {}'.format(i, params.max_iterations, cur_diff, cur_snr),
              flush=True, end='')

    return None, 0, -1


def run_attack(net, data_loader, get_feature, logger, ad_save_path, params, target):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    rng = np.random.RandomState(params.rand_seed)

    for i, (x, y) in enumerate(data_loader):
        print('sample {}/{}'.format(i + 1, len(data_loader)))
        file = data_loader.dataset.filenames[i]
        file = file if '.wav' in file else file + '.wav'

        orig_pred = net.predict(get_feature(x.to(device))).item()
        true_pred = y.item()
        if 0 <= target == orig_pred:
            print('skipping sample {}; target already prediction'.format(file))
            logger.append([file, 0, 0, False, true_pred, orig_pred, orig_pred, target.item()])
            continue

        cur_target = get_random_target(orig_pred, rng) if target == -1 else target
        ad_ex, snr, converged = target_attack(x.to(device), cur_target.to(device), net, get_feature, params)
        if converged > 0:
            print('\nsave file: {} with distortion: {}db (r:{}/o:{})'.format(file, snr, true_pred, orig_pred))
            save_adv_example(ad_ex.cpu(), os.path.join(ad_save_path, file))
            new_pred = net.predict(get_feature(ad_ex)).item()
            logger.append([file, converged, snr, True, true_pred, orig_pred, new_pred, cur_target.item()])
        else:
            print('\ncould not find robust adv. example for file {}'.format(file))
            logger.append([file, converged, snr, False, true_pred, orig_pred, orig_pred, cur_target.item()])


def main():
    # get params
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    param_file = os.path.join(cur_dir, 'attack_config.txt')
    params = get_params(param_file)
    target = check_target(params.target)

    # load model to be attacked
    net = prep_net(params.model_to_attack)
    # prep directories for experiment
    ad_save_path = prep_dir(params.experiment, get_label_from_idx(target))
    # prep data and feature computation
    get_feature, data_loader = prep_feature_data(params)
    # prep logger
    logger = prep_logger(param_file, params, get_label_from_idx(target))

    # run fgsm
    run_attack(net, data_loader, get_feature, logger, ad_save_path, params, target)


if __name__ == '__main__':
    main()
