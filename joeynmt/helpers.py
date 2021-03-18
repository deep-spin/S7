# coding: utf-8
"""
Collection of helper functions
"""
from typing import Dict, Callable, Optional, List
import copy
import glob
import os
import os.path
from os.path import join
import errno
import shutil
import random
import logging
from logging import Logger

import yaml

import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset

from joeynmt.vocabulary import Vocabulary
from joeynmt.plotting import plot_heatmap


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(
        "{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Hello! This is Joey-NMT.")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  vocabs: Dict[str, Vocabulary],
                  logging_function: Callable[[str], None]) -> None:
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param vocabs:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain %d,\n\tvalid %d,\n\ttest %d",
        len(train_data), len(valid_data),
        len(test_data) if test_data is not None else 0)

    examples = [(f, " ".join(vars(train_data[0])[f])) for f in vocabs]
    example_text = "\n".join(["\t[{}] {}".format(*ex) for ex in examples])
    logging_function("First training example:\n{}".format(example_text))

    for f, vocab in vocabs.items():
        start = "First 10 words ({}): ".format(f)
        logging_function(start + "%s", " ".join(
            '(%d) %s' % (i, t) for i, t in enumerate(vocab.itos[:10])))

    for f, vocab in vocabs.items():
        logging_function("Number of {} word types: {}".format(f, len(vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def postprocess(raw_sequences, level):
    join_char = " " if level in ["word", "bpe"] else ""
    cooked_seqs = [join_char.join(s) for s in raw_sequences]
    if level == "bpe":
        cooked_seqs = [bpe_postprocess(s) for s in cooked_seqs]
    return cooked_seqs


def store_attention_plots(attentions: dict,
                          targets: List[List[str]],
                          sources: List[List[str]],
                          model_dir: str,
                          indices: List[int],
                          tb_writer: Optional[SummaryWriter] = None,
                          steps: int = 0,
                          data_set_name: str = "att") -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    assert all(i < len(sources) for i in indices)
    for i in indices:
        for name, attn in attentions.items():
            output_prefix = join(
                model_dir, "{}.{}.{}".format(data_set_name, name, steps)
            )
            col_labels = targets if name == "trg_trg" else sources
            row_labels = sources if name == "src_src" else targets
            plot_file = "{}.{}.pdf".format(output_prefix, i)
            attn_matrix = attn[i]
            cols = col_labels[i]
            rows = row_labels[i]

            fig = plot_heatmap(scores=attn_matrix, column_labels=cols,
                               row_labels=rows, output_path=plot_file, dpi=100)
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(
                    scores=attn_matrix, column_labels=cols,
                    row_labels=rows, output_path=None, dpi=50)
                # the tensorboard thing will need fixing
                tb_writer.add_figure(
                    "attention_{}/{}.".format(name, i), fig, global_step=steps
                )


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def pad_and_stack_hyps(hyps, pad_value):
    shape = len(hyps), max(h.shape[0] for h in hyps)
    filled = torch.full(shape, pad_value, dtype=int)
    for j, h in enumerate(hyps):
        filled[j, :h.shape[0]] = h
    return filled


def pad_and_stack_3d(hyps, pad_value):
    batch_size = len(hyps)
    n_samples = len(hyps[0])
    max_len = max(max(h.shape[0] for h in beam) for beam in hyps)
    shape = batch_size, n_samples, max_len
    filled = torch.full(shape, pad_value, dtype=int)
    for i, beam in enumerate(hyps):
        for j, h in enumerate(beam):
            filled[i, j, :h.shape[0]] = h
    return filled
