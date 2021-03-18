from typing import Dict
from collections import defaultdict

import torch
from torch import Tensor

from joeynmt.helpers import pad_and_stack_hyps, tile


def greedy_search(model, encoder_output, masks, max_output_length: int,
                  labels: dict = None):
    """
    Greedy decoding. Select the token word highest probability at each time
    step.

    :param masks: mask source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attn_scores: attention scores (3d array) (rnn)
                               or None (transformer)
    """
    transformer = model.is_transformer
    any_mask = next(iter(masks.values()))
    batch_size = any_mask.size(0)

    att_vectors = None

    prev_y = any_mask.new_full(
        [batch_size, 1], model.bos_index, dtype=torch.long
    )

    output = []
    attn_scores = defaultdict(list)
    hidden = None

    masks["trg"] = any_mask.new_ones([1, 1, 1]) if transformer else None
    finished = any_mask.new_zeros((batch_size, 1)).bool()

    for t in range(max_output_length):

        # decode a single step
        logits, hidden, att_probs, att_vectors = model.decode(
            trg_input=prev_y,
            encoder_output=encoder_output,
            masks=masks,
            decoder_hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            labels=labels)
        # logits: batch x time x vocab

        # this makes it greedy: always select highest-scoring index
        next_word = torch.argmax(logits[:, -1], dim=-1).unsqueeze(1)

        if transformer:
            # transformer: keep all previous steps as input
            prev_y = torch.cat([prev_y, next_word], dim=1)
        else:
            # rnn: only the most recent step is input
            prev_y = next_word
            output.append(next_word.squeeze(1).cpu())
            for k, v in att_probs.items():
                if v is not None:
                    # currently returning None in multi-source
                    attn_scores[k].append(v.squeeze(1).cpu())

        is_eos = torch.eq(next_word, model.eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all of batch
        if (finished >= 1).sum() == batch_size:
            break

    stacked_output = prev_y[:, 1:] if transformer else torch.stack(output, dim=1)

    if transformer:
        if att_probs is not None:
            stacked_attn = {k: v.cpu() for k, v in att_probs.items()}
        else:
            stacked_attn = None
    else:
        stacked_attn = {k: torch.stack(v, axis=1) for k, v in attn_scores.items()}
    return stacked_output, stacked_attn, None


def beam_search(
        model,
        size: int,
        encoder_output,
        masks: Dict[str, Tensor],
        max_output_length: int,
        scorer,
        labels: dict = None,
        return_scores: bool = False):
    """
    Beam search with size k.

    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param encoder_output:
    :param masks:
    :param max_output_length:
    :param scorer: function for rescoring hypotheses
    :param embed:
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    transformer = model.is_transformer
    any_mask = next(iter(masks.values()))
    batch_size = any_mask.size(0)

    att_vectors = None  # not used for Transformer
    device = encoder_output.device

    if model.is_ensemble:
        # run model.ensemble_bridge, I guess
        hidden = model.ensemble_bridge(encoder_output)
    else:
        if not transformer and model.decoder.bridge_layer is not None:
            hidden = model.decoder.bridge_layer(encoder_output.hidden)
        else:
            hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        # layers x batch*k x dec_hidden_size
        if isinstance(hidden, list):
            hidden = [tile(h, size, dim=1) if h is not None else None
                      for h in hidden]
        else:
            hidden = tile(hidden, size, dim=1)

    # encoder_output: batch*k x src_len x enc_hidden_size
    encoder_output.tile(size, dim=0)
    masks = {k: tile(v, size, dim=0) for k, v in masks.items() if k != "trg"}
    masks["trg"] = any_mask.new_ones([1, 1, 1]) if transformer else None

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # beam_size copies of each batch element
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=device
    )

    # keeps track of the top beam size hypotheses to expand for each
    # element in the batch to be further decoded (that are still "alive")
    alive_seq = beam_offset.new_full((batch_size * size, 1), model.bos_index)
    prev_y = alive_seq if transformer else alive_seq[:, -1].view(-1, 1)

    # Give full probability to the first beam on the first step.
    # pylint: disable=not-callable
    current_beam = torch.tensor(
        [0.0] + [float("-inf")] * (size - 1), device=device
    ).repeat(batch_size, 1)

    results = {"predictions": [[] for _ in range(batch_size)],
               "scores": [[] for _ in range(batch_size)],
               "gold_score": [0] * batch_size}

    for step in range(1, max_output_length + 1):

        # decode a single step
        log_probs, hidden, _, att_vectors = model.decode(
            trg_input=prev_y,
            encoder_output=encoder_output,
            masks=masks,
            decoder_hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            generate="log",
            labels=labels)
        log_probs = log_probs.squeeze(1)  # log_probs: batch*k x trg_vocab

        # multiply probs by the beam probability (=add logprobs)
        raw_scores = log_probs + current_beam.view(-1).unsqueeze(1)

        # flatten log_probs into a list of possibilities
        vocab_size = log_probs.size(-1)  # vocab size
        raw_scores = raw_scores.reshape(-1, size * vocab_size)

        # apply an additional scorer, such as a length penalty
        scores = scorer(raw_scores, step) if scorer is not None else raw_scores

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = scores.topk(size, dim=-1)

        # If using a length penalty, scores are distinct from log probs.
        # The beam keeps track of log probabilities regardless
        current_beam = topk_scores if scorer is None \
            else raw_scores.gather(1, topk_ids)

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(vocab_size)
        topk_ids = topk_ids.fmod(vocab_size)

        # map beam_index to batch_index in the flat representation
        b_off = beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
        batch_index = topk_beam_index + b_off
        select_ix = batch_index.view(-1)

        # append latest prediction (result: batch_size*k x hyp_len)
        selected_alive_seq = alive_seq.index_select(0, select_ix)
        alive_seq = torch.cat([selected_alive_seq, topk_ids.view(-1, 1)], -1)

        is_finished = topk_ids.eq(model.eos_index)  # batch x beam
        if step == max_output_length:
            is_finished.fill_(1)
        top_finished = is_finished[:, 0].eq(1)  # batch

        # save finished hypotheses
        seq_len = alive_seq.size(-1)
        predictions = alive_seq.view(-1, size, seq_len)
        ix = top_finished.nonzero().view(-1)
        for i in ix:
            finished_scores = topk_scores[i]
            finished_preds = predictions[i, :, 1:]
            b = batch_offset[i]
            # if you desire more hypotheses, you can use topk/sort
            top_score, top_pred_ix = finished_scores.max(dim=0)
            top_pred = finished_preds[top_pred_ix]
            results["scores"][b].append(top_score)
            results["predictions"][b].append(top_pred)

        if top_finished.all():
            break

        # remove finished batches for the next step
        unfinished = top_finished.eq(0).nonzero().view(-1)
        current_beam = current_beam.index_select(0, unfinished)
        batch_index = batch_index.index_select(0, unfinished)
        batch_offset = batch_offset.index_select(0, unfinished)
        alive_seq = predictions.index_select(0, unfinished).view(-1, seq_len)

        # reorder indices, outputs and masks
        select_ix = batch_index.view(-1)
        encoder_output.index_select(select_ix)
        masks = {k: v.index_select(0, select_ix) if k != "trg" else v
                 for k, v in masks.items()}

        if model.is_ensemble:
            if not transformer:
                new_hidden = []
                for h_i in hidden:
                    if isinstance(h_i, tuple):
                        # for LSTMs, states are tuples of tensors
                        h, c = h_i
                        h = h.index_select(1, select_ix)
                        c = c.index_select(1, select_ix)
                        new_h_i = h, c
                    else:
                        # for GRUs, states are single tensors
                        new_h_i = h_i.index_select(1, select_ix)
                    new_hidden.append(new_h_i)
                hidden = new_hidden
        else:
            if hidden is not None and not transformer:
                if isinstance(hidden, tuple):
                    # for LSTMs, states are tuples of tensors
                    h, c = hidden
                    h = h.index_select(1, select_ix)
                    c = c.index_select(1, select_ix)
                    hidden = h, c
                else:
                    # for GRUs, states are single tensors
                    hidden = hidden.index_select(1, select_ix)

        if att_vectors is not None:
            if model.is_ensemble:
                att_vectors = [av.index_select(0, select_ix)
                               if av is not None else None
                               for av in att_vectors]
            else:
                att_vectors = att_vectors.index_select(0, select_ix)

        prev_y = alive_seq if transformer else alive_seq[:, -1].view(-1, 1)

    # is moving to cpu necessary/good?
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu() for r in results["predictions"]], model.pad_index
    )
    if return_scores:
        final_scores = torch.stack([s[0] for s in results["scores"]])
        return final_outputs, None, final_scores
    else:
        return final_outputs, None, None
