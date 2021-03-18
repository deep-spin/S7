#!/usr/bin/env python

from typing import List, Optional
from itertools import product
import numpy as np

import matplotlib
from matplotlib import rcParams
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use('Agg')


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work their way either side
    from a prescribed midpoint value)

    e.g. im=ax1.imshow(
        array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def draw_square(ax, i, j, **kwargs):
    square = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, **kwargs)
    ax.add_patch(square)


def draw_all_squares(ax, M):
    for ii in range(M.shape[0]):
        for jj in range(M.shape[1]):
            if M[ii, jj] > 0:
                draw_square(ax, jj, ii, color="#aaaaaa", lw=1, alpha=1)


def plot_heatmap(scores: np.array, column_labels: List[str],
                 row_labels: List[str], output_path: Optional[str] = None,
                 dpi: int = 300) -> Figure:

    """
    Plotting function that can be used to visualize (self-)attention.
    Plots are saved if `output_path` is specified, in format that this file
    ends with ('pdf' or 'png').

    :param scores: attention scores
    :param column_labels:  labels for columns (e.g. target tokens)
    :param row_labels: labels for rows (e.g. source tokens)
    :param output_path: path to save to
    :param dpi: set resolution for matplotlib
    :return: pyplot figure
    """
    if output_path is not None:
        assert output_path.endswith(".png") or output_path.endswith(".pdf"), \
            "output path must have .png or .pdf extension"
    assert scores.ndim == 2 or scores.ndim == 4

    # if 2d, there is only a single attention mechanism: otherwise, the first
    # two dimensions correspond to layers and heads.
    if scores.ndim == 2:
        scores = scores[None, None, :]
    # make a subplot for that (look up how this works, again)
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots.html
    n_layers, n_heads = scores.shape[:2]
    x_sent_len = len(column_labels)
    y_sent_len = len(row_labels)

    fig, axes = plt.subplots(
        nrows=n_layers, ncols=n_heads, dpi=dpi, figsize=(10, 10),
        # sharex="col", sharey="row"
    )
    coords = product(range(scores.shape[0]), range(scores.shape[1]))
    for i, j in coords:
        ax = axes[i, j] if isinstance(axes, np.ndarray) else axes
        head_scores = scores[i, j, :y_sent_len, :x_sent_len]
        # check that cut off part didn't have any attention
        assert np.sum(head_scores[y_sent_len:, :x_sent_len]) == 0

        # automatic label size
        # labelsize = 25 * (10 / max(x_sent_len, y_sent_len))

        # font config
        # rcParams['xtick.labelsize'] = labelsize
        # rcParams['ytick.labelsize'] = labelsize

        cmap = plt.cm.PuOr_r  # OrRd
        cax = ax.matshow(
            head_scores,
            cmap=cmap,
            clim=(-1, 1),
            norm=MidpointNormalize(midpoint=0, vmin=1, vmax=1))
        draw_all_squares(ax, head_scores)

        ax.xaxis.tick_top()
        ax.set_xticklabels([''] + column_labels, rotation=45,
                           horizontalalignment='left')
        ax.set_yticklabels([''] + row_labels)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()

    if output_path is not None:
        if output_path.endswith(".pdf"):
            pp = PdfPages(output_path)
            pp.savefig(fig)
            pp.close()
        else:
            plt.savefig(output_path)

    plt.close()

    return fig
