from typing import Dict

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from joeynmt.helpers import tile
from joeynmt.metrics import levenshtein_distance_cost, sentence_bleu_cost


def renormalize(log_p, alpha):
    """
    Take the model-computed log probabilities over candidates (log_p), apply
    a smoothing parameter (alpha), and renormalize to a q distribution such
    that the probabilities of candidates sum to 1. Return the log of q.
    """
    alpha_log_p = alpha * log_p
    log_q = alpha_log_p - torch.logsumexp(alpha_log_p, dim=1).unsqueeze(1)
    return log_q


def compute_costs(vocab, gold, candidates, cost, level):
    costs = {
        "levenshtein": levenshtein_distance_cost,
        "bleu": sentence_bleu_cost
    }
    cost_func = costs[cost]

    gold_toks = vocab.arrays_to_sentences(gold)  # cut at eos

    batch_size, n_samples, _ = candidates.size()  # batch x samples x len

    costs = torch.zeros(batch_size, n_samples, device=candidates.device)
    for j in range(batch_size):
        gold_seq = gold_toks[j]
        cand_seqs = candidates[j]
        cand_toks = vocab.arrays_to_sentences(cand_seqs)
        for i in range(n_samples):
            candidate_seq = cand_toks[i]
            costs[j, i] = cost_func(candidate_seq, gold_seq, level=level)

    return costs


def sample_candidates(
        model,
        n_samples: int,
        encoder_output,
        masks: Dict[str, Tensor],
        max_output_length: int,
        labels: dict = None):
    """
    Sample n_samples sequences from the model

    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param encoder_output:
    :param masks:
    :param max_output_length:
    :return:
        - stacked_output: dim?,
        - scores: dim?
    """

    # init
    transformer = model.is_transformer
    any_mask = next(iter(masks.values()))
    batch_size = any_mask.size(0)
    att_vectors = None  # not used for Transformer
    device = encoder_output.device

    masks.pop("trg", None)  # mutating one of the inputs is not good

    # Recurrent models only: initialize RNN hidden state
    if not transformer and model.decoder.bridge_layer is not None:
        hidden = model.decoder.bridge_layer(encoder_output.hidden)
    else:
        hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        # layers x batch*k x dec_hidden_size
        hidden = tile(hidden, n_samples, dim=1)

    # encoder_output: batch*k x src_len x enc_hidden_size
    encoder_output.tile(n_samples, dim=0)
    masks = {k: tile(v, n_samples, dim=0) for k, v in masks.items()}

    # Transformer only: create target mask
    masks["trg"] = any_mask.new_ones([1, 1, 1]) if transformer else None

    # the structure holding all batch_size * k partial hypotheses
    alive_seq = torch.full(
        (batch_size * n_samples, 1),
        model.bos_index,
        dtype=torch.long,
        device=device
    )
    # the structure indicating, for each hypothesis, whether it has
    # encountered eos yet (if it has, stop updating the hypothesis
    # likelihood)
    is_finished = torch.zeros(
        batch_size * n_samples, dtype=torch.bool, device=device
    )

    # for each (batch x n_samples) sequence, there is a log probability
    seq_probs = torch.zeros(batch_size * n_samples, device=device)

    for step in range(1, max_output_length + 1):
        dec_input = alive_seq if transformer \
            else alive_seq[:, -1].view(-1, 1)

        # decode a step
        probs, hidden, att_scores, att_vectors = model.decode(
            trg_input=dec_input,
            encoder_output=encoder_output,
            masks=masks,
            decoder_hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            labels=labels,
            generate="true"
        )

        # batch*k x trg_vocab
        # probs = model.decoder.gen_func(logits[:, -1], dim=-1).squeeze(1)

        next_ids = Categorical(probs).sample().unsqueeze(1)  # batch*k x 1
        next_scores = probs.gather(1, next_ids).squeeze(1)  # batch*k

        seq_probs = torch.where(
            is_finished, seq_probs, seq_probs + next_scores.log()
        )

        # append latest prediction
        # batch_size*k x hyp_len
        alive_seq = torch.cat([alive_seq, next_ids], -1)

        # update which hypotheses are finished
        is_finished = is_finished | next_ids.eq(model.eos_index).squeeze(1)

        if is_finished.all():
            break

    # final_outputs: batch x n_samples x len
    final_outputs = alive_seq.view(batch_size, n_samples, -1)
    seq_probs = seq_probs.view(batch_size, n_samples)

    return final_outputs, seq_probs


def topk_candidates(
        model,
        n_samples: int,
        encoder_output,
        masks: Dict[str, Tensor],
        max_output_length: int,
        labels: dict = None):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.
    In each decoding step, find the k most likely partial hypotheses.
    :param decoder:
    :param size: size of the beam
    :param encoder_output:
    :param masks:
    :param max_output_length:
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # size = n_best = n_samples

    # note that bos doesn't seem to be in the final hyps here

    # init
    transformer = model.is_transformer
    any_mask = next(iter(masks.values()))
    batch_size = any_mask.size(0)
    att_vectors = None  # not used for Transformer
    device = encoder_output.device

    masks.pop("trg", None)

    # Recurrent models only: initialize RNN hidden state
    if not transformer and model.decoder.bridge_layer is not None:
        hidden = model.decoder.bridge_layer(encoder_output.hidden)
    else:
        hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        # layers x batch*k x dec_hidden_size
        hidden = tile(hidden, n_samples, dim=1)

    # encoder_output: batch*k x src_len x enc_hidden_size
    encoder_output.tile(n_samples, dim=0)
    masks = {k: tile(v, n_samples, dim=0) for k, v in masks.items()}

    # Transformer only: create target mask
    masks["trg"] = any_mask.new_ones([1, 1, 1]) \
        if transformer else None

    # numbering elements in the batch
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=device
    )

    # beam_size copies of each batch element
    beam_offset = torch.arange(
        0,
        batch_size * n_samples,
        step=n_samples,
        dtype=torch.long,
        device=device)

    # keeps track of the top beam size hypotheses to expand for each
    # element in the batch to be further decoded (that are still "alive")
    # the structure holding all batch_size * k partial hypotheses
    alive_seq = torch.full(
        (batch_size * n_samples, 1),
        model.bos_index,
        dtype=torch.long,
        device=device
    )

    # the structure indicating, for each hypothesis, whether it has
    # encountered eos yet (if it has, stop updating the hypothesis
    # likelihood)

    is_finished = torch.zeros(
        batch_size * n_samples, dtype=torch.bool, device=device
    )

    # Give full probability to the first beam on the first step.
    # current_beam: batch x size
    # current_beam = torch.zeros(batch_size, n_samples, device=device)
    # seq_probs = torch.zeros(batch_size * n_samples, device=device)
    seq_probs = torch.full(
        (batch_size, n_samples), float("-inf"), device=device
    )
    seq_probs[:, 0] = 0

    for step in range(1, max_output_length + 1):
        dec_input = alive_seq if transformer \
            else alive_seq[:, -1].view(-1, 1)

        # decode a step
        logits, hidden, att_scores, att_vectors = model.decode(
            trg_input=dec_input,
            encoder_output=encoder_output,
            masks=masks,
            decoder_hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            labels=labels
        )

        # batch*k x trg_vocab
        probs = model.decoder.gen_func(logits[:, -1], dim=-1).squeeze(1)
        support = probs > 0
        support_sizes = support.sum(dim=-1)

        # multiply probs by the beam probability (=add logprobs)
        # scores: (batch*k) x trg_vocab
        past_scores = seq_probs.view(-1).unsqueeze(1)
        scores = past_scores + probs.log()
        '''
        scores = torch.where(
            support, past_scores + probs.log(), torch.tensor(float("-inf"))
        )
        '''
        # hmm, should the where come here or after selecting topk?
        '''
        scores = torch.where(
            is_finished, past_scores, past_scores + probs.log()
        )
        '''

        # "flattened" scores: batch x k*trg_vocab
        scores = scores.reshape(-1, n_samples * model.decoder.output_size)
        # so, we need to make sure that none of the

        # pick currently best top k hypotheses (each is batch x k)
        topk_scores, topk_ids = scores.topk(n_samples, dim=-1)
        seq_probs = topk_scores

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(model.decoder.output_size)
        topk_ids = topk_ids.fmod(model.decoder.output_size)

        # map beam_index to batch_index in the flat representation
        # b_off = beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
        # batch_index = topk_beam_index + b_off
        # select_ix = batch_index.view(-1)

        # append latest prediction
        # batch_size*k x hyp_len
        alive_seq = torch.cat([alive_seq, topk_ids.view(-1, 1)], -1)

        # is_finished is alive_batch x beam width
        # is_finished = topk_ids.eq(model.eos_index)
        # this is not a sparsity-friendly condition: I should check if
        # all probability mass is in the beam
        is_finished = is_finished | topk_ids.eq(model.eos_index).view(-1)
        if step == max_output_length:
            is_finished.fill_(1)

        log_beam_mass = torch.logsumexp(seq_probs, dim=-1)
        nothing_pruned = log_beam_mass == 0
        # if there is full probability in the beam and every nonzero
        # hypothesis has reached eos, you can break
        print(log_beam_mass)

        if is_finished.all():
            break

    print()
    # so, now alive_seq is (batch*n_samples) x length
    final_outputs = alive_seq.view(batch_size, n_samples, -1)

    # we must return batch x n_samples x length
    # final_outputs = pad_and_stack_3d(pred_seqs, model.pad_index)
    # supported = seq_probs.exp().gt(0)
    # print(supported.sum(dim=-1))
    print(torch.logsumexp(seq_probs, dim=-1))

    # still too much mass in the beam
    beam_mass = seq_probs.exp().sum(dim=-1)
    if beam_mass.gt(1.001).any():
        print(beam_mass)

    return final_outputs, seq_probs


def mbr_search(
        model,
        size: int,
        encoder_output,
        masks,
        max_output_length: int,
        labels: dict = None,
        cost: str = "levenshtein"):
    """
    minimum bayes risk decoding, an alternative decision rule for people sick
    of badly approximating MAP with beam search
    """
    candidates, log_p = sample_candidates(
        model,
        encoder_output=encoder_output,
        masks=masks,
        max_output_length=max_output_length,
        n_samples=size,
        labels=labels
    )
    alpha = 1.0
    candidates = candidates[:, :, 1:]  # trim bos
    log_q = renormalize(log_p, alpha)
    q = log_q.exp()  # batch x n_samples

    all_costs = []
    for s in range(candidates.size(1)):
        single_candidate = candidates[:, s, :]
        single_cost = compute_costs(model.trg_vocab, single_candidate, candidates, cost, level="word")
        all_costs.append(single_cost)
    costs = torch.stack(all_costs, dim=-1) # batch x n_samples x n_samples
    # so, do we use the q distribution here?
    risks = (costs @ q.unsqueeze(2)).squeeze(2)  # batch x n_samples
    return
