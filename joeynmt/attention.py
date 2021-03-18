# coding: utf-8
"""
Attention modules
"""

from typing import List, Dict
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from entmax import Entmax15, EntmaxBisect

from joeynmt.better_sparsemax import BetterSparsemax as Sparsemax


class AlphaChooser(nn.Module):

    def __init__(self, dim):
        super(AlphaChooser, self).__init__()
        self.chooser = nn.Linear(dim, 1)

    def forward(self, X):
        return torch.clamp(torch.sigmoid(self.chooser(X)) + 1, min=1.01)


class AttentionMechanism(nn.Module):
    """
    Base attention class
    """
    def __init__(self, hidden_size: int, attn_func: str = "softmax",
                 attn_alpha: float = 1.5):
        super(AttentionMechanism, self).__init__()
        Entmax = partial(EntmaxBisect, alpha=attn_alpha, n_iter=30)
        attn_funcs = {"softmax": nn.Softmax,
                      "sparsemax": Sparsemax,
                      "entmax15": Entmax15,
                      "entmax": Entmax}
        assert attn_func in attn_funcs, "Unknown attention function"
        self.transform = attn_funcs[attn_func](dim=-1)
        self.proj_keys = None   # to store projected keys

    @property
    def _query_dim(self):
        raise NotImplementedError

    def _check_input_shapes(self, query: Tensor, mask: Tensor, values: Tensor):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        query_batch, query_tgt_len, query_dim = query.size()
        values_batch, values_src_len, values_dim = values.size()
        mask_batch, mask_tgt_len, mask_src_len = mask.size()

        assert query_batch == values_batch == mask_batch
        assert query_tgt_len == 1 == mask_tgt_len
        assert query_dim == self._query_dim
        assert values_dim == self.key_layer.in_features
        assert mask_src_len == values_src_len

    def score(self, query, keys):
        raise NotImplementedError

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys and assign them to `self.proj_keys`.
        This pre-computation is efficiently done for all keys
        before receiving individual queries.

        :param keys: shape (batch_size, src_length, encoder.hidden_size)
        """
        # proj_keys: batch x src_len x hidden_size
        self.proj_keys = self.key_layer(keys)

    # pylint: disable=arguments-differ
    def forward(self, query: Tensor, masks: Tensor, values: Tensor):
        """
        attention forward pass.
        Computes context vectors and attention scores for a given query and
        all masked values and returns them.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, src_length)
        :param values: values (encoder states),
            shape (batch_size, src_length, encoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        assert isinstance(masks, Tensor)
        mask = masks
        self._check_input_shapes(query=query, mask=mask, values=values)

        assert self.proj_keys is not None, \
            "projection keys must be pre-computed"
        keys = self.proj_keys

        # scores: batch_size x 1 x src_length
        scores = self.score(query, keys)

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = self.transform(scores)  # batch x 1 x src_len

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x values_size

        return context, alphas


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention

    Section A.1.2 in https://arxiv.org/pdf/1409.0473.pdf.
    """

    def __init__(self, hidden_size: int, key_size: int, query_size: int,
                 **kwargs):
        """
        Creates attention mechanism.

        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """
        super(BahdanauAttention, self).__init__(hidden_size, **kwargs)

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def score(self, query, keys):
        proj_query = self.query_layer(query)

        # keys: batch x src_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(proj_query + keys))
        # scores: batch x src_len x 1

        scores = scores.squeeze(2).unsqueeze(1)  # why not transpose?
        # scores: batch x 1 x time
        return scores

    @property
    def _query_dim(self):
        return self.query_layer.in_features

    def __repr__(self):
        return "BahdanauAttention"


class LuongAttention(AttentionMechanism):
    """
    Implements Luong (bilinear / multiplicative) attention.

    Eq. 8 ("general") in http://aclweb.org/anthology/D15-1166.
    """

    def __init__(self, hidden_size: int, key_size: int, **kwargs):
        """
        Creates attention mechanism.

        :param hidden_size: size of the key projection layer, has to be equal
            to decoder hidden size
        :param key_size: size of the attention input keys
        """
        super(LuongAttention, self).__init__(hidden_size, **kwargs)
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)

    def score(self, query, keys):
        return query @ keys.transpose(1, 2)

    @property
    def _query_dim(self):
        return self.key_layer.out_features

    def __repr__(self):
        return "LuongAttention"


class MultiAttention(nn.Module):
    """Not to be confused with MultiHeadedAttention"""

    def __init__(self, attn_type: str, head_names: List[str],
                 key_size: int, hidden_size: int,
                 attn_merge: str = "concat", gate_func: str = "softmax",
                 gate_alpha: float = 1.5, attn_func: str = "softmax",
                 attn_alpha: float = 1.5, **kwargs):
        super(MultiAttention, self).__init__()
        assert attn_type in ["luong", "bahdanau"]
        assert isinstance(attn_func, str) or isinstance(attn_func, dict)
        assert isinstance(attn_alpha, float) or isinstance(attn_alpha, dict)
        attn = LuongAttention if attn_type == "luong" else BahdanauAttention
        self.head_names = head_names
        if isinstance(attn_func, str):
            attn_func = {k: attn_func for k in head_names}
        if isinstance(attn_alpha, float):
            attn_alpha = {k: attn_alpha for k in head_names}
        self.heads = nn.ModuleDict(
            {k: attn(key_size=key_size, hidden_size=hidden_size,
                     attn_func=attn_func[k], attn_alpha=attn_alpha[k],
                     **kwargs)
             for k in head_names}
        )

        assert attn_merge in ["concat", "gate"]
        if attn_merge == "concat":
            self.attn_merge = AttentionConcat(
                enc_size=key_size,
                hidden_size=hidden_size,
                head_names=self.head_names)
        else:
            self.attn_merge = AttentionGate(
                enc_size=key_size,
                hidden_size=hidden_size,
                head_names=self.head_names,
                gate_func=gate_func,
                gate_alpha=gate_alpha
            )

    @property
    def _query_dim(self):
        return sum(attn._query_dim for attn in self.heads.values())

    def forward(self, query: Tensor, masks: Dict[str, Tensor],
                values: Dict[str, Tensor]):

        contexts = dict()
        att_probs = dict()
        for head_name in self.head_names:
            head_values = values[head_name]
            mask = masks[head_name]
            context, alphas = self.heads[head_name](
                query=query, masks=mask, values=head_values
            )
            contexts[head_name] = context
            att_probs[head_name] = alphas
        att_vector = self.attn_merge(query, contexts)

        return att_vector, att_probs

    def compute_proj_keys(self, keys: Dict[str, Tensor]):
        """
        Compute the projection of the keys and assign them to `self.proj_keys`.
        This pre-computation is efficiently done for all keys
        before receiving individual queries.

        :param keys: shape (batch_size, src_length, encoder.hidden_size)
        """
        for head in self.heads:
            self.heads[head].proj_keys = self.heads[head].key_layer(keys[head])


class AttentionConcat(nn.Module):
    def __init__(self, enc_size: int, hidden_size, head_names: list):
        super(AttentionConcat, self).__init__()
        self.head_names = head_names
        input_size = hidden_size + enc_size * len(head_names)
        attn_vector_layer = nn.Linear(input_size, hidden_size)
        self.concat = nn.Sequential(attn_vector_layer, nn.Tanh())

    def forward(self, query: Tensor, contexts: Dict[str, Tensor]):
        context_list = [contexts[k] for k in self.head_names]
        input = torch.cat([query] + context_list, dim=2)
        return self.concat(input)


class AttentionGate(nn.Module):

    def __init__(self, enc_size: int, hidden_size, head_names: list,
                 gate_func: str = "softmax", gate_alpha: float = 1.5):
        super(AttentionGate, self).__init__()

        self.head_names = head_names

        Entmax = partial(EntmaxBisect, alpha=gate_alpha, n_iter=30)
        gate_funcs = {"softmax": nn.Softmax,
                      "sparsemax": Sparsemax,
                      "entmax15": Entmax15,
                      "entmax": Entmax}
        assert gate_func in gate_funcs, "Unknown gate function"
        transform = gate_funcs[gate_func](dim=-1)

        # now that I think about it, I'm not wild about this way of doing gates
        gate_input_dim = enc_size * len(head_names) + hidden_size
        n_classes = len(head_names)

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, n_classes), transform
        )

        self.candidates = nn.ModuleDict()
        for name in head_names:
            W_u = nn.Linear(enc_size + hidden_size, hidden_size)
            layer = nn.Sequential(W_u, nn.Tanh())
            self.candidates[name] = layer

    def forward(self, query: Tensor, contexts: Dict[str, Tensor]):
        """
        query: batch x 1 x hidden_size
        contexts: values are batch x 1 x enc_sizes[name]
        """
        context_list = [contexts[name] for name in self.head_names]
        # for the gate, we need to make gate_input: batch x 1 x sum(sizes)
        gate_input = torch.cat([query] + context_list, dim=2)
        gate_weights = self.gate(gate_input)  # batch x 1 x n_classes

        candidate_states = {name: torch.cat([query, context], dim=2)
                            for name, context in contexts.items()}
        candidate_vecs = {name: self.candidates[name](state)
                          for name, state in candidate_states.items()}

        candidates = torch.cat(
            [candidate_vecs[name] for name in self.head_names], dim=1
        )

        gated_context = gate_weights @ candidates
        return gated_context
