# coding: utf-8

"""
Various decoders
"""
from functools import partial
from typing import Dict
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, MultiAttention
from joeynmt.helpers import freeze_params, subsequent_mask
from joeynmt.transformer_layers import PositionalEncoding, \
    TransformerDecoderLayer
from entmax import entmax15, sparsemax, entmax_bisect


def log_sparsemax(*args, **kwargs):
    return torch.log(sparsemax(*args, **kwargs))


def log_entmax15(*args, **kwargs):
    return torch.log(entmax15(*args, **kwargs))


def log_entmax(*args, **kwargs):
    return torch.log(entmax_bisect(*args, **kwargs))


class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=0, activation="tanh", merge="cat",
                 lstm=False, decoder_layers=1):
        super(Bridge, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = repeat(hidden_size)
        self._hidden_size = hidden_size
        activations = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "none": None
        }
        assert activation in activations
        activation_func = activations[activation]
        if num_layers > 0:
            layers = []
            for inp, out in zip(input_sizes, output_sizes):
                linear = nn.Linear(inp, out)
                layers.append(linear)
                if activation_func is not None:
                    layers.append(activation_func)
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None

        assert merge in ["cat", "sum", "max"]
        self.merge = merge
        self.lstm = lstm
        self.decoder_layers = decoder_layers

    def forward(self, encoder_final):
        if isinstance(encoder_final, dict):
            # do you cat, sum, or max-pool?
            enc_finals = [encoder_final[k] for k in sorted(encoder_final)]
            if self.merge == "cat":
                hidden = torch.cat(enc_finals, dim=-1)
            else:
                hidden = torch.stack(enc_finals)
                if self.merge == "sum":
                    hidden = hidden.sum(dim=0)
                else:
                    hidden = hidden.max(dim=0)[0]
        else:
            hidden = encoder_final

        enc_hidden_size = hidden.size(1)

        if self.mlp is not None:
            hidden = self.mlp(hidden)  # batch x hidden_size
        else:
            # "last" case. Is the encoder bidirectional?
            if enc_hidden_size == 2 * self._hidden_size:
                hidden = hidden[:, :self._hidden_size]

        # expand to n_layers x batch_size x hidden_size
        # The same hidden is used for all layers
        hidden = hidden.unsqueeze(0).repeat(self.decoder_layers, 1, 1)

        if self.lstm:
            hidden = hidden, hidden
        return hidden


# pylint: disable=abstract-method
class Decoder(nn.Module):
    """
    Base decoder class
    """
    def __init__(
        self, hidden_size, vocab_size, emb_dropout, gen_func: str = "softmax",
        gen_alpha: float = 1.5, output_bias=False
    ):
        super(Decoder, self).__init__()

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(
            hidden_size, vocab_size, bias=output_bias
        )

        gen_funcs = {
            "softmax": F.softmax,
            "sparsemax": sparsemax,
            "entmax15": entmax15,
            "entmax": partial(entmax_bisect, alpha=gen_alpha, n_iter=30)}
        log_gen_funcs = {
            "softmax": F.log_softmax,
            "sparsemax": log_sparsemax,
            "entmax15": log_entmax15,
            "entmax": partial(log_entmax, alpha=gen_alpha, n_iter=30)}
        assert gen_func in gen_funcs
        assert gen_func in log_gen_funcs
        self.gen_func = gen_funcs[gen_func]
        self.log_gen_func = log_gen_funcs[gen_func]

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self.output_layer.out_features


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class RecurrentDecoder(Decoder):
    """A conditional RNN decoder with attention."""

    def __init__(self,
                 rnn_type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder_output_size: int = 0,
                 attention: str = "bahdanau",
                 num_layers: int = 1,
                 vocab_size: int = 0,
                 dropout: float = 0.,
                 emb_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 init_hidden: str = "bridge",
                 input_feeding: bool = True,
                 freeze: bool = False,
                 attn_func: str = "softmax",
                 attn_alpha: float = 1.5,
                 gen_func: str = "softmax",
                 gen_alpha: float = 1.5,
                 output_bias: bool = False,
                 multi_source: bool = False,
                 head_names: list = None,
                 attn_merge: str = "gate",
                 gate_func: str = "softmax",
                 gate_alpha: float = 1.5,
                 **kwargs) -> None:
        """
        Create a recurrent decoder with attention.

        :param rnn_type: rnn type, valid options: "lstm", "gru"
        :param emb_size: target embedding size
        :param hidden_size: size of the RNN
        :param encoder_output_size:
        :param attention: type of attention, valid options: "bahdanau", "luong"
        :param num_layers: number of recurrent layers
        :param vocab_size: target vocabulary size
        :param hidden_dropout: applied to the input to the attentional layer.
        :param dropout: Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param init_hidden: If "bridge" (default), the decoder hidden states
            are initialized from a projection of the last encoder state,
            if "zeros" they are initialized with zeros,
            if "last" they are identical to the last encoder state
            (only if they have the same size)
        :param input_feeding: Use Luong's input feeding.
        :param freeze: Freeze the parameters of the decoder during training.
        :param kwargs:
        """

        super(RecurrentDecoder, self).__init__(
            hidden_size,
            vocab_size,
            emb_dropout,
            gen_func=gen_func,
            gen_alpha=gen_alpha,
            output_bias=output_bias)

        self.multi_source = multi_source

        self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        input_size = emb_size + hidden_size if input_feeding else emb_size

        # the decoder RNN
        self.rnn = rnn(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.)

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder_output_size, hidden_size, bias=True)

        assert attention in ["bahdanau", "luong"], \
            "Unknown attention mechanism: %s. Use 'bahdanau' or 'luong'."

        if multi_source:
            attn_mechanism = partial(
                MultiAttention,
                attn_type=attention,
                head_names=head_names,
                attn_merge=attn_merge,
                gate_func=gate_func,
                gate_alpha=gate_alpha)

        elif attention == "luong":
            attn_mechanism = LuongAttention
        else:
            attn_mechanism = BahdanauAttention

        if attention == "bahdanau":
            attn_mechanism = partial(attn_mechanism, query_size=hidden_size)

        self.attention = attn_mechanism(
            hidden_size=hidden_size,
            key_size=encoder_output_size,
            attn_func=attn_func,
            attn_alpha=attn_alpha)

        # init_hidden: "bridge", "zero", "last", or a dictionary describing
        # an arbitrary-layered MLP
        assert isinstance(init_hidden, dict) or isinstance(init_hidden, str), \
            '''
            Specify either a shortcut name ("bridge", "zero", "last") or a
            dictionary containing a configuration for the bridge layer.
            '''
        if init_hidden == "zero":
            self.bridge_layer = None  # easy-peasy
        else:
            if init_hidden == "last":
                # not actually clear to me if this is necessary
                assert encoder_output_size in {hidden_size, 2 * hidden_size}, \
                    "Mismatched hidden sizes (enc: {}, dec: {})".format(
                        encoder_output_size, hidden_size
                    )
            if isinstance(init_hidden, str):
                bridge = init_hidden == "bridge"
                # 'bridge' and 'last' are shortcuts to specific special cases
                init_hidden = {
                    "num_layers": 1 if bridge else 0,
                    "activation": "tanh" if bridge else "none",
                    "merge": "cat"
                }

            if init_hidden["merge"] == "cat":
                n_heads = len(head_names) if head_names is not None else 1
                bridge_in_size = encoder_output_size * n_heads  # for cat
            else:
                bridge_in_size = encoder_output_size

            self.bridge_layer = Bridge(
                bridge_in_size,
                hidden_size,
                lstm=isinstance(self.rnn, nn.LSTM),
                decoder_layers=self.num_layers,
                **init_hidden
            )

        if freeze:
            freeze_params(self)

    @property
    def rnn_input_size(self):
        return self.rnn.input_size

    @property
    def num_layers(self):
        return self.rnn.num_layers

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    def _check_shapes_input_forward_step(self,
                                         prev_embed: Tensor,
                                         prev_att_vector: Tensor,
                                         encoder_states: Tensor,
                                         masks: Dict[str, Tensor],
                                         hidden: Tensor) -> None:
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_states:
        :param src_mask:
        :param hidden:
        """
        if isinstance(masks, dict):
            src_masks = {k: v for k, v in masks.items() if k != "trg"}
        else:
            src_masks = {"src": masks}
        batch_size = prev_embed.shape[0]
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size([1, self.hidden_size])
        assert prev_att_vector.shape[0] == batch_size

        assert all(mask.shape[0] == batch_size for mask in src_masks.values())
        assert all(mask.shape[1] == 1 for mask in src_masks.values())

        if isinstance(hidden, tuple):  # for lstm
            hidden = hidden[0]
        if hidden is not None:
            hidden_layers, hidden_batch, hidden_size = hidden.shape
            assert hidden_layers == self.num_layers
            assert hidden_batch == batch_size
            assert hidden_size == self.hidden_size

    def _check_shapes_input_forward(self,
                                    trg_embed: Tensor,
                                    encoder_output,
                                    masks: Dict[str, Tensor],
                                    hidden: Tensor = None,
                                    prev_att_vector: Tensor = None) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param trg_embed:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        :param prev_att_vector:
        """
        if isinstance(masks, dict):
            src_masks = {k: v for k, v in masks.items() if k != "trg"}
        else:
            src_masks = {"src": masks}
        batch_size, trg_len, trg_emb_size = trg_embed.shape

        assert all(mask.shape[1] == 1 for mask in src_masks.values())
        assert all(mask.shape[0] == batch_size for mask in src_masks.values())

        assert encoder_output.batch_size == batch_size

        if isinstance(encoder_output.states, dict):
            assert all(f_name in masks for f_name in encoder_output.states)
            seq_lens = encoder_output.seq_len
            assert(src_masks[k].shape[2] == v for k, v in seq_lens.items())
        else:
            assert src_masks["src"].shape[2] == encoder_output.seq_len

        assert trg_emb_size == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):  # for lstm
                hidden = hidden[0]
            assert hidden.shape[1] == batch_size
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == batch_size
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(self,
                      prev_embed: Tensor,
                      prev_att_vector: Tensor,  # context or att vector
                      encoder_states: Tensor,  # or a dict
                      masks: Dict[str, Tensor],
                      hidden: Tensor) -> (Tensor, Tensor, Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
            shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
            shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
            shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
            shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
            shape (num_layers, batch_size, hidden_size)
        :return:
            - att_vector: new attention vector (batch_size, 1, hidden_size),
            - hidden: new hidden state with shape (batch_size, 1, hidden_size),
            - att_probs: attention probabilities (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(
            prev_embed=prev_embed,
            prev_att_vector=prev_att_vector,
            encoder_states=encoder_states,
            masks=masks,
            hidden=hidden)

        if self.input_feeding:
            # concatenate the input with the previous attention vector
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        # compute context vector using attention mechanism
        # only use last layer for attention mechanism
        # key projections are pre-computed
        context, att_probs = self.attention(
            query=query, values=encoder_states, masks=masks)

        if self.multi_source:
            att_vector = self.hidden_dropout(context)
        else:
            # return attention vector (Luong)
            # combine context with decoder hidden state before prediction
            att_vector_input = torch.cat([query, context], dim=2)
            # batch x 1 x 2*enc_size+hidden_size
            att_vector_input = self.hidden_dropout(att_vector_input)

            att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs

    def forward(self,
                trg_embed: Tensor,
                encoder_output,  # an EncoderOutput
                masks: Dict[str, Tensor],
                unroll_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None,
                **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
         Unroll the decoder one step at a time for `unroll_steps` steps.
         For every step, the `_forward_step` function is called internally.

         During training, the target inputs (`trg_embed') are already known for
         the full sequence, so the full unrol is done.
         In this case, `hidden` and `prev_att_vector` are None.

         For inference, this function is called with one step at a time since
         embedded targets are the predictions from the previous time step.
         In this case, `hidden` and `prev_att_vector` are fed from the output
         of the previous call of this function (from the 2nd step on).

         `src_mask` is needed to mask out the areas of the encoder states that
         should not receive any attention,
         which is everything after the first <eos>.

         The `encoder_output` are the hidden states from the encoder and are
         used as context for the attention.

         The `encoder_hidden` is the last encoder hidden state that is used to
         initialize the first hidden decoder state (if not using zeroes or a
         bridge)

        :param trg_embed: emdedded target inputs,
            shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states and last state from the encoder,
            shapes (batch_size, src_length, encoder.output_size)
            and (batch_size x encoder.output_size)
        :param masks: one per encoded sequence: 0s for padded areas,
            1s for the rest, shape (batch_size, 1, src_length)
        :param unroll_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
            if not given it's initialized as in `self.init_hidden`,
            shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
            if not given it's initialized with zeros,
            shape (batch_size, 1, hidden_size)
        :return:
            - outputs: shape (batch_size, unroll_steps, vocab_size),
            - hidden: last hidden state (num_layers, batch_size, hidden_size),
            - att_probs: attention probabilities
                with shape (batch_size, unroll_steps, src_length),
            - att_vectors: attentional vectors
                with shape (batch_size, unroll_steps, hidden_size)
        """

        # shape checks
        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            masks=masks,
            hidden=hidden,
            prev_att_vector=prev_att_vector)

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None and self.bridge_layer is not None:
            hidden = self.bridge_layer(encoder_output.hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=encoder_output.states)

        # here we store intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = trg_embed.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = trg_embed.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_states=encoder_output.states,
                masks=masks,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        if self.multi_source:
            # future work: save probs from multi-encoder
            att_probs = None
        else:
            att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unroll_steps, src_length
        outputs = self.output_layer(att_vectors)
        # outputs: batch, unroll_steps, vocab_size
        return outputs, hidden, {"src_trg": att_probs}, att_vectors

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 self_attn_func: str = "softmax",
                 src_attn_func: str = "softmax",
                 self_attn_alpha: float = 1.5,
                 src_attn_alpha: float = 1.5,
                 gen_func: str = "softmax",
                 gen_alpha: float = 1.5,
                 output_bias: bool = False,
                 **kwargs):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__(
            hidden_size,
            vocab_size,
            emb_dropout,
            gen_func=gen_func,
            gen_alpha=gen_alpha,
            output_bias=output_bias)

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                self_attn_func=self_attn_func,
                self_attn_alpha=self_attn_alpha,
                src_attn_func=src_attn_func,
                src_attn_alpha=src_attn_alpha)
             for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        if freeze:
            freeze_params(self)

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def num_heads(self):
        return self.layers[0].trg_trg_att.num_heads

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output=None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :return:
            context attn probs: batch x layer x head x tgt x src
        """
        assert trg_mask is not None, "trg_mask is required for Transformer"

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(
                x=x, memory=encoder_output.states,
                src_mask=src_mask, trg_mask=trg_mask
            )

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, self.num_layers, self.num_heads)
