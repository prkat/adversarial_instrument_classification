# based on https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr
# -gallery-images-contours-and-fields-image-annotated-heatmap-py
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from adversarial_ml.utils.paths import misc_path
from adversarial_ml.evaluation.evaluation_utils import *
from adversarial_ml.data.data_utils import get_unique_label_set


def plot_confusion_matrix(plot_name, log_file_path):
    """ Plots confusion matrix based on log_file_path. """
    labels, confusion_matrix = _log_confusion_matrix(log_file_path)
    labels = [label.replace('_', ' ') for label in labels]
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    im = ax.imshow(confusion_matrix, cmap='PuBu', vmin=0., vmax=1.)

    ax.figure.colorbar(im, ax=ax)

    # labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('True labels')
    ax.set_ylabel('Predicted labels')
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-40, ha='right', rotation_mode='anchor')

    grid_lines = np.arange(len(labels) + 1) - .5
    ax.set_xticks(grid_lines, minor=True)
    ax.set_yticks(grid_lines, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_axisbelow(True)

    for edge, spine in ax.spines.items():
        spine.set_color('white')
        spine.set_linewidth(4)

    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle(np.array([i, i]) - 0.5, 1, 1, facecolor='none',
                                   edgecolor='darkorange', linewidth=3))

    threshold = im.norm(confusion_matrix.max()) / 2.
    kw = dict(horizontalalignment='center', verticalalignment='center')
    textcolors = ('black', 'white')
    valfmt = ticker.StrMethodFormatter('{x:.2f}')
    for i in range(len(labels)):
        for j in range(len(labels)):
            kw.update(color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])
            im.axes.text(j, i, valfmt(confusion_matrix[i, j], None), **kw)

    plt.subplots_adjust(top=.75, bottom=.03, left=.3, right=.95)
    plt.savefig(os.path.join(misc_path, plot_name))
    plt.close()


def _log_confusion_matrix(log_file_path):
    """ Compute confusion matrix based on log-file. """
    with open(log_file_path, 'r') as fp:
        log = [line.rstrip() for line in fp]

    indices = np.array(log[0].split(','))
    log = np.array([line.split(',') for line in log[1:]])
    true_idx = np.argmax(indices == 'true')
    new_idx = np.argmax(indices == 'new')

    labels = np.array(sorted(get_unique_label_set()))
    confusion_matrix = np.zeros((len(labels), len(labels)))

    for entry in log:
        confusion_matrix[int(entry[new_idx]), int(entry[true_idx])] += 1

    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=0)
    return labels, confusion_matrix


if __name__ == '__main__':
    plot_confusion_matrix('test.png', os.path.join(misc_path, 'logs/test_torch16s1f.csv'))
