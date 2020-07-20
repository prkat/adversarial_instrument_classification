import os

import numpy as np
import parse
import torch

import torchaudio
import attrdict as attr

from adversarial_ml.utils.paths import model_path


def read_file(path, filename):
    """ Given path and name of a file, returns content as list. """
    with open(os.path.join(path, filename), 'r') as fp:
        result = [f.rstrip() for f in fp]
    return result


def get_params(file_path='models/params.txt'):
    """ Reads and returns dictionary of defined (model) parameters. """
    params = attr.AttrDict({})
    with open(file_path, 'r') as fp:
        lines = [l.rstrip().rsplit('=') for l in fp if l.rstrip() and not l.startswith('#')]
    for k, v in lines:
        k, v = k.rstrip(), v.strip()
        try:
            # int
            v = int(v)
        except ValueError:
            try:
                # float
                v = float(v)
            except ValueError:
                # string or None/True/False
                if 'None' in v:
                    v = None
                elif 'True' in v:
                    v = True
                elif 'False' in v:
                    v = False
        params.update({k: v})

    return params


def load_net(model_name, model, epoch=-1):
    """ Given initialised network architecture and possibly epoch, returns network prepped for evaluation. """
    _, net, _ = _load_checkpoint(model, os.path.join(model_path, model_name), epoch=epoch, optimizer=None)
    net.eval()
    return net


def save_adv_example(adv_ex, file_path):
    """ Saves defined wave-file with sampling rate 16kHz. """
    adv_ex = adv_ex.detach()
    torchaudio.save(file_path, adv_ex, sample_rate=16000)


def _load_checkpoint(net, path, epoch=-1, optimizer=None):
    """ Loads CNN from checkpoint. """
    if '~' in path:
        path = os.path.expanduser(path)

    # is epoch == -1, load latest checkpoint
    if epoch == -1:
        files = os.listdir(path)
        files = [file for file in files if 'model_ep' in file]
        if len(files) < 1:
            raise ValueError('Cannot load a checkpoint; None yet exists at this path!')
        epoch = np.max(np.array([int(parse.parse('model_ep{}.tar', model)[0]) for model in files]))

    file_path = os.path.join(path, 'model_ep{}.tar').format(epoch)
    if not os.path.exists(file_path):
        raise ValueError('Cannot load this checkpoint; Path does not exist!')

    if torch.cuda.is_available():
        checkpoint = torch.load(file_path, map_location='cuda:0')
    else:
        checkpoint = torch.load(file_path, map_location='cpu')

    epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return epoch, net, optimizer
