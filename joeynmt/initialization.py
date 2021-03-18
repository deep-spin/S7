# coding: utf-8

"""
Implements custom initialization
"""

import math
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out, xavier_uniform_, \
    uniform_, normal_, zeros_


def orthogonal_rnn_init_(cell: nn.RNNBase, gain: float = 1.):
    """
    Orthogonal initialization of recurrent weights
    RNN parameters contain 3 or 4 matrices in one parameter, so we slice it.
    """
    with torch.no_grad():
        for _, hh, _, _ in cell.all_weights:
            for i in range(0, hh.size(0), cell.hidden_size):
                nn.init.orthogonal_(hh.data[i:i + cell.hidden_size], gain=gain)


def lstm_forget_gate_init_(cell: nn.RNNBase, value: float = 1.) -> None:
    """
    Initialize LSTM forget gates with `value`.

    :param cell: LSTM cell
    :param value: initial value, default: 1
    """
    with torch.no_grad():
        for _, _, ih_b, hh_b in cell.all_weights:
            l = len(ih_b)
            ih_b.data[l // 4:l // 2].fill_(value)
            hh_b.data[l // 4:l // 2].fill_(value)


def xavier_uniform_n_(w: Tensor, gain: float = 1., n: int = 4) -> None:
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.

    :param w: parameter
    :param gain: default 1
    :param n: default 4
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def _parse_init(s, scale, _gain):
    scale = float(scale)
    assert scale > 0, "incorrect init_weight"
    init_fns = {
        "xavier": partial(xavier_uniform_, gain=_gain),
        "uniform": partial(uniform_, a=-scale, b=scale),
        "normal": partial(normal_, mean=0., std=scale),
        "zeros": zeros_}
    s = s.lower()
    assert s in init_fns, "unknown initializer"
    return init_fns[s]


# pylint: disable=too-many-branches
def initialize_model(model: nn.Module, cfg: dict, padding_idx: int) -> None:
    """
    This initializes a model based on the provided config.

    All initializer configuration is part of the `model` section of the
    configuration file.
    For an example, see e.g. `https://github.com/joeynmt/joeynmt/
    blob/master/configs/iwslt_envi_xnmt.yaml#L47`

    The main initializer is set using the `initializer` key.
    Possible values are `xavier`, `uniform`, `normal` or `zeros`.
    (`xavier` is the default).

    When an initializer is set to `uniform`, then `init_weight` sets the
    range for the values (-init_weight, init_weight).

    When an initializer is set to `normal`, then `init_weight` sets the
    standard deviation for the weights (with mean 0).

    The word embedding initializer is set using `embed_initializer` and takes
    the same values. The default is `normal` with `embed_init_weight = 0.01`.

    Biases are initialized separately using `bias_initializer`.
    The default is `zeros`, but you can use the same initializers as
    the main initializer.

    Set `init_rnn_orthogonal` to True if you want RNN orthogonal initialization
    (for recurrent matrices). Default is False.

    `lstm_forget_gate` controls how the LSTM forget gate is initialized.
    Default is `1`.

    :param model: model to initialize
    :param cfg: the model configuration
    :param padding_idx:
    """

    # defaults: xavier, embeddings: normal 0.01, biases: zeros, no orthogonal
    gain = float(cfg.get("init_gain", 1.0))  # for xavier
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))

    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    embed_gain = float(cfg.get("embed_init_gain", 1.0))  # for xavier

    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "embed" in name:
                embed_init_fn_(p)
                # zero out paddings; assumes all fields have same pad
                p.data[padding_idx].zero_()

            elif "bias" in name:
                if bias_init != "xavier":
                    bias_init_fn_(p)

            elif len(p.size()) > 1:

                # RNNs combine multiple matrices is one, which messes up
                # xavier initialization
                if init == "xavier" and "rnn" in name:
                    if "encoders" in name:
                        rnn = next(iter(model.encoders.values())).rnn
                    elif "encoder" in name:  # matches "encoders" too...
                        rnn = model.encoder.rnn
                    elif "decoder" in name:
                        rnn = model.decoder.rnn
                    else:
                        rnn = None

                    if rnn is not None:
                        n = 4 if isinstance(rnn, nn.LSTM) else 3
                    else:
                        n = 1  # when would this come up?
                    xavier_uniform_n_(p.data, gain=gain, n=n)
                else:
                    init_fn_(p)

        orthogonal = cfg.get("init_rnn_orthogonal", False)
        lstm_forget_gate = cfg.get("lstm_forget_gate", 1.)

        # encoder rnn orthogonal initialization & LSTM forget gate
        if hasattr(model, "encoders"):
            encoders = list(model.encoders.values())
        else:
            encoders = [model.encoder]
        for encoder in encoders:
            if hasattr(encoder, "rnn"):

                if orthogonal:
                    orthogonal_rnn_init_(encoder.rnn)

                if isinstance(encoder.rnn, nn.LSTM):
                    lstm_forget_gate_init_(encoder.rnn, lstm_forget_gate)

        # decoder rnn orthogonal initialization & LSTM forget gate
        if hasattr(model.decoder, "rnn"):

            if orthogonal:
                orthogonal_rnn_init_(model.decoder.rnn)

            if isinstance(model.decoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.decoder.rnn, lstm_forget_gate)
