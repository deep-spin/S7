from typing import Dict

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from joeynmt.helpers import tile


def sample_decode(
        model,
        size: int,
        encoder_output,
        masks: Dict[str, Tensor],
        max_output_length: int,
        labels: dict = None):
    """
    Sample size sequences from the model

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
        hidden = tile(hidden, size, dim=1)

    # encoder_output: batch*k x src_len x enc_hidden_size
    encoder_output.tile(size, dim=0)
    masks = {k: tile(v, size, dim=0) for k, v in masks.items()}

    # Transformer only: create target mask
    masks["trg"] = any_mask.new_ones([1, 1, 1]) if transformer else None

    # the structure holding all batch_size * k partial hypotheses
    alive_seq = torch.full(
        (batch_size * size, 1),
        model.bos_index,
        dtype=torch.long,
        device=device
    )
    # the structure indicating, for each hypothesis, whether it has
    # encountered eos yet (if it has, stop updating the hypothesis
    # likelihood)
    is_finished = torch.zeros(
        batch_size * size, dtype=torch.bool, device=device
    )

    # for each (batch x size) sequence, there is a log probability
    seq_probs = torch.zeros(batch_size * size, device=device)

    for step in range(1, max_output_length + 1):
        dec_input = alive_seq if transformer else alive_seq[:, -1].view(-1, 1)

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

    # final_outputs: batch x size x len
    final_outputs = alive_seq.view(batch_size, size, -1)
    seq_probs = seq_probs.view(batch_size, size)
    max_scores, max_ix = seq_probs.max(dim=-1)
    outs = []
    for b in range(final_outputs.size(0)):
        outs.append(final_outputs[b, max_ix[b]])
    best_outputs = torch.stack(outs)  # hmm, maybe not as good as pad and stack
    # print(torch.index_select(final_outputs, 0, max_ix).size())
    #print(final_outputs[:, max_ix].size())
    #print(final_outputs[:, max_ix].size())

    return best_outputs, max_scores
