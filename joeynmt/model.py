# coding: utf-8
"""
Module to represents whole models
"""

from typing import Dict, List
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings, FeatureEmbeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch
from joeynmt.helpers import ConfigurationError, tile
from joeynmt.search import beam_search, greedy_search
from joeynmt.sample_decoding import sample_decode
from joeynmt.exact_search import depth_first_search, full_traversal, \
    iterative_deepening, single_queue_decode
from joeynmt.minimum_risk import sample_candidates, topk_candidates, \
    renormalize, compute_costs


class EncoderOutput:
    def __init__(self, encoder_states: Tensor, encoder_hidden: Tensor):
        assert len(encoder_states.shape) == 3
        assert encoder_hidden is None or len(encoder_hidden.shape) == 2
        if encoder_hidden is not None:
            assert encoder_hidden.shape[-1] == encoder_states.shape[-1]
        self.states = encoder_states  # instead of encoder_output
        self.hidden = encoder_hidden

    def tile(self, size, dim):
        self.states = tile(self.states.contiguous(), size, dim=dim)

    def index_select(self, select_ix):
        self.states = self.states.index_select(0, select_ix)

    def split(self):
        """
        splits a single EncoderOutput object into batch_size objects.
        This is useful for decoding techniques that do not work batchwise,
        as the sequence can still be encoded in batches.
        """
        states = self.states.split(1)
        if self.hidden is not None:
            hiddens = self.hidden.split(1)
        else:
            hiddens = [None] * len(states)
        assert len(states) == len(hiddens) == self.batch_size
        return [EncoderOutput(s, h) for (s, h) in zip(states, hiddens)]

    @property
    def device(self):
        return self.states.device

    @property
    def batch_size(self):
        return self.states.size(0)

    @property
    def seq_len(self):
        return self.states.size(1)


class MultiEncoderOutput:
    def __init__(self, encoder_states: Dict[str, Tensor],
                 encoder_hidden: Dict[str, Tensor]):
        assert all(len(v.shape) == 3 for v in encoder_states.values())
        assert all(len(v.shape) == 2 for v in encoder_hidden.values())
        if encoder_hidden is not None:
            # make sure the sizes match up
            assert all(encoder_states[k].size(-1) == encoder_hidden[k].size(-1)
                       for k in encoder_states)

        batch_size = next(iter(encoder_states.values())).size(0)
        assert all(v.size(0) == batch_size for v in encoder_states.values())
        self.states = encoder_states  # instead of encoder_output
        self.hidden = encoder_hidden
        # note that not all of the state sequences will be the same length

    def tile(self, size, dim):
        self.states = {k: tile(v.contiguous(), size, dim=dim)
                       for k, v in self.states.items()}

    def index_select(self, select_ix):
        self.states = {k: v.index_select(0, select_ix)
                       for k, v in self.states.items()}

    @property
    def device(self):
        return next(iter(self.states.values())).device

    @property
    def batch_size(self):
        return next(iter(self.states.values())).size(0)

    @property
    def seq_len(self):
        return {k: v.size(1) for k, v in self.states.items()}


class EnsembleEncoderOutput:
    def __init__(self, encoder_outputs: List[EncoderOutput]):
        self._encoder_outputs = encoder_outputs

    @property
    def states(self):
        return [eo.states for eo in self._encoder_outputs]

    @property
    def hidden(self):
        return [eo.hidden for eo in self._encoder_outputs]

    def tile(self, size, dim):
        for eo in self._encoder_outputs:
            eo.tile(size, dim)

    def index_select(self, select_ix):
        for eo in self._encoder_outputs:
            eo.index_select(select_ix)

    @property
    def device(self):
        return self._encoder_outputs[0].device

    @property
    def batch_size(self):
        return self._encoder_outputs[0].batch_size

    @property
    def seq_len(self):
        return self._encoder_outputs[0].seq_len

    @property
    def outs(self):
        return self._encoder_outputs


class _Model(nn.Module):

    def encode(self, batch):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def _generate(self, logits, single_step: bool, log: bool):
        gen_func = self.decoder.log_gen_func if log else self.decoder.gen_func
        if single_step:
            logits = logits[:, -1]
        probs = gen_func(logits, dim=-1)
        if single_step:
            logits = logits.squeeze(1)
        return probs

    @property
    def is_transformer(self):
        return isinstance(self.decoder, TransformerDecoder)

    @property
    def is_ensemble(self):
        return isinstance(self, EnsembleModel)

    # pylint: disable=arguments-differ
    def forward(self, batch) -> (Tensor, Tensor, Tensor, Tensor):
        """
        :param batch: a batch with all relevant fields
        :return: decoder outputs
        """
        encoder_output, _ = self.encode(batch)
        '''
        trg_input = batch["trg"][0][:, :-1]
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           masks=batch.masks,
                           labels=batch.labels)
        '''
        return self.force_decode(batch, encoder_output)

    def force_decode(self, batch, encoder_output):
        trg_input = batch["trg"][0][:, :-1]
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           masks=batch.masks,
                           labels=batch.labels)

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module,
                           encoder_output=None):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable
        if encoder_output is not None:
            out, hidden, att_probs, _ = self.force_decode(batch, encoder_output)
        else:
            out, hidden, att_probs, _ = self(batch)

        # compute log probs
        out = out.view(-1, out.size(-1))

        # compute batch loss
        gold = batch["trg"][0][:, 1:].contiguous().view(-1)
        batch_loss = loss_function(out, gold)
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

    def get_risk_for_batch(
            self, batch: Batch, n_samples: int, alpha: float,
            strategy: str, max_len: int, cost: str, level: str) -> Tensor:
        """
        Minimum-risk seq2seq training
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # what are the gold sequences?
        gold = batch["trg"][0][:, 1:]  # batch x gold_len

        assert strategy in ["sample", "topk"]
        encoder_output, _ = self.encode(batch)

        # compute candidate sequences from the model and their log probs
        # candidates: batch x n_samples x length
        generate = topk_candidates if strategy == "topk" else sample_candidates
        candidates, log_p = generate(
            self,
            encoder_output=encoder_output,
            masks=batch.masks,
            max_output_length=max_len,
            n_samples=n_samples,
            labels=batch.labels
        )
        candidates = candidates[:, :, 1:]  # trim bos
        log_q = renormalize(log_p, alpha)
        q = log_q.exp()

        costs = compute_costs(self.trg_vocab, gold, candidates, cost, level)

        # compute expectation of cost
        batch_loss = (costs * q).sum()

        return batch_loss

    def get_loss_and_risk_for_batch(
            self, batch: Batch, loss_function: nn.Module,
            n_samples: int, alpha: float, strategy: str, max_len: int,
            cost: str, level: str, mrt_lambda: float) -> Tensor:
        """
        Minimum-risk seq2seq training
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """

        assert strategy in ["sample", "topk"]
        encoder_output, _ = self.encode(batch)

        trg_input = batch["trg"][0][:, :-1]
        unroll_steps = trg_input.size(1)
        out = self.decode(
            encoder_output=encoder_output,
            trg_input=trg_input,
            unroll_steps=unroll_steps,
            masks=batch.masks,
            labels=batch.labels)[0]
        mle_out = out.view(-1, out.size(-1))
        flattened_gold = batch["trg"][0][:, 1:].contiguous().view(-1)
        mle_loss = loss_function(mle_out, flattened_gold)

        gold = batch["trg"][0][:, 1:]  # batch x gold_len

        # compute candidate sequences from the model and their log probs
        # candidates: batch x n_samples x length
        generate = topk_candidates if strategy == "topk" else sample_candidates
        candidates, log_p = generate(
            self,
            encoder_output=encoder_output,
            masks=batch.masks,
            max_output_length=max_len,
            n_samples=n_samples,
            labels=batch.labels
        )
        candidates = candidates[:, :, 1:]  # trim bos
        log_q = renormalize(log_p, alpha)
        q = log_q.exp()

        costs = compute_costs(self.trg_vocab, gold, candidates, cost, level)

        # compute expectation of cost
        mrt_loss = (costs * q).sum()

        return mle_loss + mrt_lambda * mrt_loss

    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  scorer, method=None, encoder_output=None,
                  return_scores=False, max_hyps=1, break_at_p=1.0,
                  break_at_argmax=False):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param scorer: scoring function for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        with torch.no_grad():
            if encoder_output is None:
                encoder_output, enc_attn = self.encode(batch)
            else:
                enc_attn = None

            # currently max_output_length will only work if there is a "src"
            # field
            if max_output_length is None and "src" in batch:
                # adapt to src length
                max_output_length = int(batch["src"][1].max().item() * 1.5)

            if method is None:
                # normal stuff
                if beam_size == 0:
                    search = greedy_search
                else:
                    search = partial(
                        beam_search, size=beam_size, scorer=scorer,
                        return_scores=return_scores
                    )
                stacked_output, dec_attn, beam_scores = search(
                    self,
                    encoder_output=encoder_output,
                    masks=batch.masks,
                    max_output_length=max_output_length,
                    labels=batch.labels
                )

            else:
                # abnormal stuff
                dec_attn = None
                assert method in {"dfs", "dfs_enumerate", "bfs_enumerate",
                                  "iddfs", "sqd", "sample"}
                if method == "sample":
                    stacked_output, beam_scores = sample_decode(
                        self,
                        encoder_output=encoder_output,
                        masks=batch.masks,
                        max_output_length=max_output_length,
                        labels=batch.labels,
                        size=beam_size
                    )
                elif method == "sqd":
                    stacked_output, beam_scores = single_queue_decode(
                        self,
                        encoder_output=encoder_output,
                        masks=batch.masks,
                        max_output_length=max_output_length,
                        labels=batch.labels,
                        buffer_size=beam_size,
                        max_hyps=max_hyps
                    )
                elif method == "iddfs":
                    stacked_output, beam_scores = iterative_deepening(
                        self,
                        encoder_output=encoder_output,
                        masks=batch.masks,
                        max_output_length=max_output_length,
                        labels=batch.labels,
                        buffer_size=beam_size,
                        max_hyps=max_hyps
                    )
                elif method == "dfs":
                    stacked_output, beam_scores = depth_first_search(
                        self,
                        encoder_output=encoder_output,
                        masks=batch.masks,
                        max_output_length=max_output_length,
                        labels=batch.labels
                    )
                else:
                    # this method terminates when it reaches a certain number
                    # of finished hypotheses
                    stacked_output, beam_scores = full_traversal(
                        self,
                        encoder_output=encoder_output,
                        masks=batch.masks,
                        max_output_length=max_output_length,
                        labels=batch.labels,
                        max_hyps=max_hyps,
                        mode="dfs" if method == "dfs_enumerate" else "bfs",
                        break_at_p=break_at_p,
                        break_at_argmax=break_at_argmax
                    )

            attn = dict()
            if enc_attn is not None:
                for k, v in enc_attn.items():
                    attn[k] = v.cpu().numpy()
            if dec_attn is not None:
                for k, v in dec_attn.items():
                    attn[k] = v

            return stacked_output, attn, beam_scores


class Model(_Model):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    def encode(self, batch) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        f_name, (src, src_lengths, src_mask) = next(batch.items())
        assert f_name == "src"
        src_emb = self.src_embed(src, **batch.labels)
        enc_states, enc_hidden, enc_attn = self.encoder(
            src_emb, src_lengths, src_mask)
        return EncoderOutput(enc_states, enc_hidden), enc_attn

    def decode(self, encoder_output: Tensor, masks: Dict[str, Tensor],
               trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               labels: dict = None,
               generate: str = None,
               **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states, and last encoder state
        :param smask: masks, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        assert generate is None or generate in ["log", "true"]
        src_mask = masks["src"]
        trg_mask = masks.get("trg", None)
        if labels is None:
            labels = dict()

        dec_out, hidden, att_probs, prev_att_vector = self.decoder(
            trg_embed=self.trg_embed(trg_input, **labels),
            encoder_output=encoder_output,
            masks=src_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            trg_mask=trg_mask,
            **kwargs)

        if generate is not None:
            dec_out = self._generate(
                dec_out, single_step=unroll_steps == 1, log=generate == "log"
            )
        return dec_out, hidden, att_probs, prev_att_vector

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed,
                                    self.trg_embed)


class MultiSourceModel(_Model):

    def __init__(self,
                 encoders: Dict[str, Encoder],
                 decoder: Decoder,
                 embeds: Dict[str, Embeddings],
                 vocabs: Dict[str, Vocabulary]):
        """
        Create a multi-source seq2seq model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        assert set(encoders) <= set(embeds)
        assert set(embeds) <= set(vocabs)
        assert "trg" in embeds
        super(MultiSourceModel, self).__init__()

        self.embeds = nn.ModuleDict(embeds)
        self.encoders = nn.ModuleDict(encoders)
        self.decoder = decoder
        self.vocabs = vocabs
        self.bos_index = vocabs["trg"].stoi[BOS_TOKEN]
        self.pad_index = vocabs["trg"].stoi[PAD_TOKEN]
        self.eos_index = vocabs["trg"].stoi[EOS_TOKEN]

    @property
    def trg_vocab(self):
        return self.vocabs["trg"]

    def encode(self, batch):
        """
        Encode a batch with fields for multiple encoders.
        """
        states = dict()
        hiddens = dict()
        for f_name, value in batch.items():
            seq, lengths, mask = value
            emb_matrix = self.embeds[f_name]
            encoder = self.encoders[f_name]
            emb = emb_matrix(seq, **batch.labels)
            enc_states, enc_hidden, _ = encoder(emb, lengths, mask)
            states[f_name] = enc_states
            hiddens[f_name] = enc_hidden
        output = MultiEncoderOutput(states, hiddens)

        return output, None

    def decode(self,
               encoder_output: MultiEncoderOutput,
               masks: Dict[str, Tensor],
               trg_input: Tensor,
               unroll_steps: int,
               decoder_hidden: Tensor = None,
               labels: dict = None,
               generate: str = None,
               **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_outputs: encoder states for attention computation and
            decoder initialization
        :param masks: each is 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        trg_embed = self.embeds["trg"](trg_input, **labels)
        gen_funcs = {"log": self.decoder.log_gen_func,
                     "true": self.decoder.gen_func}
        assert generate is None or generate in gen_funcs
        if labels is None:
            labels = dict()
        trg_embed = self.embeds["trg"](trg_input, **labels)
        dec_out, hidden, att_probs, prev_att_vector = self.decoder(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            masks=masks,
            **kwargs)

        if generate is not None:
            dec_out = self._generate(
                dec_out, single_step=unroll_steps == 1, log=generate == "log"
            )

        return dec_out, hidden, att_probs, prev_att_vector

    def __repr__(self) -> str:
        return "{}(\n\tencoders={},\n\tdecoder={})".format(
            self.__class__.__name__, self.encoders, self.decoder
        )


class EnsembleModel(_Model):

    def __init__(self, *models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        assert all(m.trg_vocab == self.trg_vocab for m in self.models)
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    @property
    def trg_vocab(self):
        return self.models[0].trg_vocab

    def encode(self, batch):
        """
        Encode a batch with fields for multiple encoders.
        """
        encoder_outputs = []
        for m in self.models:
            enc_out, _ = m.encode(batch)
            encoder_outputs.append(enc_out)

        return EnsembleEncoderOutput(encoder_outputs), None

    @property
    def is_transformer(self):
        return any(m.is_transformer for m in self.models)

    def ensemble_bridge(self, encoder_output):
        # gotta make a...list?...of hiddens
        hiddens = []
        for m, eo in zip(self.models, encoder_output.outs):
            if not m.is_transformer and m.decoder.bridge_layer is not None:
                hidden = m.decoder.bridge_layer(eo.hidden)
            else:
                hidden = None
            hiddens.append(hidden)
        return hiddens

    def decode(self,
               encoder_output: EnsembleEncoderOutput,
               masks: Dict[str, Tensor],
               trg_input: Tensor,
               unroll_steps: int,
               prev_att_vector: list = None,
               decoder_hidden: list = None,
               generate: str = "true",
               **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
        assert generate in ["log", "true"]
        models = self.models
        assert len(models) == len(encoder_output.outs)

        if prev_att_vector is not None:
            assert len(prev_att_vector) == len(models)
        else:
            prev_att_vector = [None] * len(models)
        if decoder_hidden is not None:
            assert len(decoder_hidden) == len(models)
        else:
            decoder_hidden = [None] * len(models)

        single_decodes = []
        inputs = zip(models, prev_att_vector,
                     decoder_hidden, encoder_output.outs)
        for model, prev_att, dec_hid, enc_out in inputs:
            single_decode = model.decode(
                encoder_output=enc_out,
                masks=masks,
                trg_input=trg_input,
                unroll_steps=unroll_steps,
                decoder_hidden=dec_hid,
                prev_att_vector=prev_att,
                generate="true",
                **kwargs
            )
            single_decodes.append(single_decode)

        dec_outs = [d[0] for d in single_decodes]
        hiddens = [d[1] for d in single_decodes]
        att_vectors = [d[3] for d in single_decodes]

        mean_probs = torch.stack(dec_outs).mean(dim=0).unsqueeze(1)
        out_probs = mean_probs if generate == "true" else torch.log(mean_probs)
        return out_probs, hiddens, {"src_trg": None}, att_vectors


def build_embeddings(emb_config: dict, vocab: Vocabulary):
    padding_idx = vocab.stoi[PAD_TOKEN]

    embed = Embeddings(
        **emb_config, vocab_size=len(vocab), padding_idx=padding_idx
    )
    return embed


def build_feature_embeddings(emb_configs: dict, vocabs: dict, main: str):
    assert set(emb_configs) <= set(vocabs)
    assert main in emb_configs
    embs = {k: build_embeddings(emb_configs[k], vocabs[k])
            for k in emb_configs}
    main_emb = embs.pop(main)
    return FeatureEmbeddings(main_emb, mode="feature", **embs)


def build_model(cfg: dict = None, vocabs: dict = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    if "encoders" in cfg:
        # two cases: are columns provided? If so, make an identical encoder
        # for each of them.
        # If instead keys are given...
        if "columns" in cfg["encoders"]:
            enc_columns = cfg["encoders"]["columns"]
            assert all(column in vocabs for column in enc_columns)
            shared_cfg = cfg["encoders"]["encoder"]
            enc_configs = {column: shared_cfg for column in enc_columns}
            share_embs = cfg["encoders"].get("share_embeddings", False)
            share_encoders = cfg["encoders"].get("share_encoders", False)
            if share_embs:
                any_v = next(v for k, v in vocabs.items() if k != "trg")
                assert all(v == any_v for k, v in vocabs.items() if k != "trg")
        else:
            enc_columns = list(cfg["encoders"].keys())
            enc_configs = cfg["encoders"]
            share_embs = False
            share_encoders = False
    else:
        enc_columns = ["src"]
        enc_configs = {"src": cfg["encoder"]}
        share_embs = False
        share_encoders = False

    dec_config = cfg["decoder"]

    emb_configs = {name: enc_config["embeddings"]
                   for name, enc_config in enc_configs.items()}

    emb_configs["trg"] = dec_config["embeddings"]

    embeds = dict()
    encoders = dict()
    for enc_column, enc_cfg in enc_configs.items():
        # make each encoder

        if "feature_embeddings" in enc_cfg:
            # feature embeddings features come from label fields of a tsv
            embed = build_feature_embeddings(
                enc_cfg["feature_embeddings"], vocabs, enc_column)
        else:
            if share_embs and embeds:
                # get something that's already in the dict
                embed = next(iter(embeds.values()))
            else:
                # make a new embedding matrix
                vocab = vocabs[enc_column]
                emb_cfg = enc_cfg["embeddings"]
                embed = Embeddings(
                    **emb_cfg,
                    vocab_size=len(vocab),
                    padding_idx=vocab.stoi[PAD_TOKEN]
                )
        embeds[enc_column] = embed

        if share_encoders and encoders:
            encoder = next(iter(encoders.values()))
        else:
            enc_dropout = enc_cfg.get("dropout", 0.)
            enc_emb_dropout = enc_cfg["embeddings"].get("dropout", enc_dropout)
            enc_type = enc_cfg.get("type", "recurrent")
            '''
            if enc_type == "transformer":
                enc_emb_size = emb_cfg["embedding_dim"]
                enc_hidden_size = enc_cfg["hidden_size"]
                assert enc_emb_size == enc_hidden_size, \
                    "for transformer, emb_size must be hidden_size"
            '''
            enc_class = TransformerEncoder if enc_type == "transformer" \
                else RecurrentEncoder
            encoder = enc_class(
                **enc_cfg,
                emb_size=embed.embedding_dim,
                emb_dropout=enc_emb_dropout)
        encoders[enc_column] = encoder

    trg_vocab = vocabs["trg"]

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        assert vocabs["src"].itos == vocabs["trg"].itos, \
            "Embedding cannot be tied because vocabularies differ."
        embeds["trg"] = embeds["src"]
    else:
        # build the target embeddings
        if "feature_embeddings" in dec_config:
            # feature embeddings features come from label fields of a tsv
            embed = build_feature_embeddings(
                dec_config["feature_embeddings"], vocabs, "trg")
        else:
            trg_vocab = vocabs["trg"]
            dec_emb_cfg = dec_config["embeddings"]
            embed = Embeddings(
                **dec_emb_cfg,
                vocab_size=len(trg_vocab),
                padding_idx=trg_vocab.stoi[PAD_TOKEN]
            )
        embeds["trg"] = embed

    # build decoder
    dec_dropout = dec_config.get("dropout", 0.)
    dec_type = dec_config.get("type", "recurrent")
    dec_class = TransformerDecoder if dec_type == "transformer" \
        else RecurrentDecoder
    decoder = dec_class(
        **dec_config,
        encoder_output_size=encoder.output_size,
        vocab_size=len(vocabs["trg"]),
        emb_size=embeds["trg"].embedding_dim,
        emb_dropout=emb_configs["trg"].get("dropout", dec_dropout),
        multi_source=len(encoders) > 1,
        head_names=list(encoders.keys()))

    if len(encoders) == 1:
        model = Model(
            encoder=encoders["src"],
            decoder=decoder,
            src_embed=embeds["src"],
            trg_embed=embeds["trg"],
            src_vocab=vocabs["src"],
            trg_vocab=vocabs["trg"])
    else:
        model = MultiSourceModel(
            encoders=encoders,
            decoder=decoder,
            embeds=embeds,
            vocabs=vocabs
        )

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if embeds["trg"].lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = embeds["trg"].lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, vocabs["trg"].stoi[PAD_TOKEN])

    return model
