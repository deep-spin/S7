import math
import heapq

import torch

from joeynmt.helpers import pad_and_stack_hyps


def depth_first_search(model, encoder_output, masks, max_output_length,
                       labels: dict = None):
    # non-recursive wrapper around the recursive function that does DFS
    enc_outs = encoder_output.split()
    ys = []
    gammas = []
    for i, enc_out in enumerate(enc_outs):
        seq_masks = {k: v[i].unsqueeze(0) for k, v in masks.items()}
        if model.is_transformer:
            seq_masks["trg"] = torch.ones(1, 1, 1, dtype=torch.bool, device=masks["trg"].device)
        #print(seq_masks["trg"].size())
        if labels is not None:
            seq_labels = {k: v[i].unsqueeze(0) for k, v in labels.items()}
        else:
            seq_labels = None
        y, gamma = _depth_first_search(
            model,
            enc_out,
            seq_masks,
            max_output_length + 1,  # fixing fencepost?
            labels=seq_labels)
        y = y[1:] if y is not None else None  # necessary?
        ys.append(y)
        gammas.append(gamma)

    outputs = pad_and_stack_hyps(ys, model.pad_index)
    # any_finished = outputs.eq(3).any(dim=-1)

    scores = torch.FloatTensor(gammas, device=outputs.device)
    # assert scores[~any_finished].eq(float("-inf")).all()

    return outputs, scores


def iterative_deepening(model, encoder_output, masks, max_output_length,
                        labels: dict = None, buffer_size: int = 1, max_hyps=1,
                        verbose=False):
    enc_outs = encoder_output.split()
    ys = []
    gammas = []
    for i, enc_out in enumerate(enc_outs):
        best_y = None
        gamma = float("-inf")
        for max_len in range(max_output_length):
            if verbose:
                print("new max length", max_len)
            seq_masks = {k: v[i].unsqueeze(0) for k, v in masks.items()}
            if model.is_transformer:
                seq_masks["trg"] = torch.ones(1, 1, 1, dtype=torch.bool, device=masks["trg"].device)
            if labels is not None:
                seq_labels = {k: v[i].unsqueeze(0) for k, v in labels.items()}
            hypotheses = _sqd(
                model,
                enc_out,
                masks=seq_masks,
                max_output_length=max_len,
                gamma=gamma,
                labels=seq_labels,
                buffer_size=buffer_size
            )
            for j in range(max_hyps):
                try:
                    p, y = next(hypotheses)
                except StopIteration:
                    break
                if p > gamma:
                    gamma = p
                    best_y = y
                    if verbose:
                        y_seq = [model.trg_vocab.itos[y_i] for y_i in best_y]
                        print(gamma, y_seq)
            if verbose:
                print()
        ys.append(best_y)
        gammas.append(gamma)

    outputs = pad_and_stack_hyps(ys, model.pad_index)
    # any_finished = outputs.eq(3).any(dim=-1)

    scores = torch.FloatTensor(gammas, device=outputs.device)
    # assert scores[~any_finished].eq(float("-inf")).all()

    return outputs, scores


def _depth_first_search(
        model, encoder_output, masks, depth, y=None,
        p=0.0, gamma=float('-inf'), decoder_hidden=None, att_vectors=None,
        labels: dict = None
):
    """
    An exact depth-first search for NMT (see Stahlberg & Byrne, 2019)
    (plus a little bit about sorting the hypotheses and stopping expansion
    when they fall below the threshold)
    """
    device = encoder_output.device
    transformer = model.is_transformer
    # shouldn't need masks: the batch size should be 1
    if y is None:
        y = model.bos_index,  # a tuple
        # empty sequence case
        # y = torch.LongTensor([model.bos_index])  # 1

    # is the hypothesis finished?
    if y[-1] == model.eos_index:
        # if finished, return y and its probability
        # make y into a tensor on the right device
        return torch.tensor(y, device=device), p

    # have you reached the depth limit without a completed hypothesis?
    if depth == 0:
        # note how this tensor will not end with eos
        return torch.tensor(y, device=device), float('-inf')

    # compute probabilities
    if transformer:
        prev_y = torch.tensor(y, device=device).unsqueeze(0)
    else:
        prev_y = torch.tensor([y[-1]], device=device).view(1, 1)
    log_p, decoder_hidden, _, att_vectors = model.decode(
        trg_input=prev_y,
        encoder_output=encoder_output,
        masks=masks,
        decoder_hidden=decoder_hidden,
        prev_att_vector=att_vectors,
        unroll_steps=1,
        labels=labels,
        generate="log"
    )
    p_prime = log_p.view(-1) + p
    nonzeros = p_prime.gt(float("-inf")).nonzero().view(-1)
    possible = sorted(
        [(i, p_prime[i]) for i in nonzeros], key=lambda x: x[1], reverse=True
    )

    y_tilde = None  # will be replaced by the thing we return
    for i, p_prime_i in possible:
        if p_prime_i < gamma:
            break  # right?
        # for RNN; transformer requires whole seq
        y_prime, gamma_prime = _depth_first_search(
            model,
            encoder_output,
            masks,
            depth - 1 if depth is not None else None,
            y=y + (i,),
            p=p_prime_i.item(),  # hmm
            gamma=gamma,
            att_vectors=att_vectors,
            decoder_hidden=decoder_hidden,
            labels=labels
        )
        if gamma_prime > gamma:
            y_tilde, gamma = y_prime, gamma_prime
        elif y_tilde is None and y_prime is not None:
            y_tilde = y_prime
    '''
    if y_tilde is None:
        y_tilde = y_prime
    '''

    return y_tilde, gamma


class SearchState:
    def __init__(self, y, decoder_hidden, att_vectors):
        self.y = y
        self.decoder_hidden = decoder_hidden
        self.att_vectors = att_vectors

    def __eq__(self, other):
        return self.y == other.y

    def __ne__(self, other):
        return self.y != other.y

    def __lt__(self, other):
        return self.y < other.y

    def __gt__(self, other):
        return self.y > other.y

    def __le__(self, other):
        return self.y <= other.y

    def __ge__(self, other):
        return self.y >= other.y


class PQueue:
    # why? Because I don't like the heapq interface
    def __init__(self, maxheap=True):
        self.maxheap = maxheap
        self.heap = []

    def pop(self):
        score, data = heapq.heappop(self.heap)
        if self.maxheap:
            score = -score
        return score, data

    def append(self, item):
        score, data = item
        if self.maxheap:
            score = -score
        heapq.heappush(self.heap, (score, data))

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return bool(self.heap)


# will need to make a small change here
def _traverse(model, encoder_output, masks, max_output_length,
              mode="dfs", labels: dict = None, gamma=float("-inf"),
              prune=False):
    """
    Lazily generate hypotheses from the model, with options that control the
    order in which hypotheses are generated (currently: depth-first, sorted)
    """
    device = encoder_output.device
    transformer = model.is_transformer
    sparse = model.decoder.gen_func.__name__ != "softmax"

    bos_state = 0.0, SearchState((model.bos_index,), None, None)
    hyps = [] if mode == "dfs" else PQueue()
    hyps.append(bos_state)
    while hyps:
        current_p, current_hyp = hyps.pop()
        if current_hyp.y[-1] == model.eos_index:
            gamma = max(gamma, current_p)
            # yield the hypothesis
            yield current_p, torch.tensor(current_hyp.y[1:], device=device)
        elif len(current_hyp.y) - 2 < max_output_length and (not prune or current_p >= gamma):
            # I don't think the current_p >= gamma test works with bfs
            # generate successors and add them to the stack
            # the current_p >= gamma thing might be part of the problem
            if transformer:
                prev_y = torch.tensor(current_hyp.y, device=device).unsqueeze(0)
            else:
                prev_y = torch.tensor([current_hyp.y[-1]], device=device).view(1, 1)
            log_p, decoder_hidden, _, att_vectors = model.decode(
                trg_input=prev_y,
                encoder_output=encoder_output,
                masks=masks,
                decoder_hidden=current_hyp.decoder_hidden,
                prev_att_vector=current_hyp.att_vectors,
                unroll_steps=1,
                labels=labels,
                generate="log"
            )
            p_prime = log_p.view(-1) + current_p

            if sparse:
                nonzeros = p_prime.gt(float("-inf")).nonzero().view(-1)
                successors = ((i, p_prime[i]) for i in nonzeros)
            else:
                successors = enumerate(p_prime)

            if prune:
                successors = [(i, p_i) for (i, p_i) in successors if p_i >= gamma]
            else:
                successors = list(successors)
            # pruned = [(i, p_i) for (i, p_i) in successors if p_i >= gamma]
            # print(len(successors), len(pruned))
            # pruned = successors
            possible = sorted(successors, key=lambda x: x[1])
            # note that all of the possibilities are the same length.
            # if only it were easier to batch
            for i, p in possible:
                p = p_prime[i]
                new_state = p, SearchState(current_hyp.y + (i,), decoder_hidden, att_vectors)
                hyps.append(new_state)


def full_traversal(
        model, encoder_output, masks, max_output_length, max_hyps=1,
        mode="dfs", labels: dict = None, break_at_argmax=False, break_at_p=1.0):
    assert mode in ["dfs", "bfs"]

    enc_outs = encoder_output.split()
    ys = []
    gammas = []
    n_hyps = []
    for i, enc_out in enumerate(enc_outs):
        seq_masks = {k: v[i].unsqueeze(0) for k, v in masks.items()}
        if model.is_transformer:
            seq_masks["trg"] = torch.ones(1, 1, 1, dtype=torch.bool, device=masks["trg"].device)
        #print(seq_masks["trg"].size())
        if labels is not None:
            seq_labels = {k: v[i].unsqueeze(0) for k, v in labels.items()}
        hypotheses = _traverse(model, enc_out, seq_masks, max_output_length,
                               mode=mode, labels=seq_labels, prune=break_at_argmax)
        total_p = 0.0
        current_best = float("-inf")
        generated = []
        for j in range(max_hyps):
            try:
                p, y = next(hypotheses)
                # print(p, [model.trg_vocab.itos[y_i] for y_i in y])
            except StopIteration:
                break
            generated.append((p, y))
            total_p += math.exp(p)
            if p > current_best:
                current_best = p
            if total_p > break_at_p or (break_at_argmax and current_best > 1 - total_p):
                # either you've found the argmax or you've found a hypothesis set
                # that covers a very large part of the mass
                break
        best_p, best_y = max(generated) if generated else (float("-inf"), (model.bos_index,))
        ys.append(best_y)
        gammas.append(best_p)
        n_hyps.append(len(generated))
        print(total_p, len(generated))

    outputs = pad_and_stack_hyps(ys, model.pad_index)
    # any_finished = outputs.eq(3).any(dim=-1)

    scores = torch.FloatTensor(gammas, device=outputs.device)
    # assert scores[~any_finished].eq(float("-inf")).all()

    # idea: return the size of the set of hypotheses

    return outputs, scores


def _mini_traversal(
        model, encoder_output, masks, max_output_length, max_hyps=1,
        mode="dfs", labels: dict = None, break_at_argmax=False, gamma=float("-inf")):
    assert mode in ["dfs", "bfs"]

    hypotheses = _traverse(model, encoder_output, masks, max_output_length,
                           mode=mode, labels=labels)
    # total_p = 0.0
    best_y = None
    for j in range(max_hyps):
        try:
            p, y = next(hypotheses)
        except StopIteration:
            break
        if p > gamma:
            gamma = p
            best_y = y
        '''
        if break_at_argmax and gamma > 1 - total_p:
            # the long tail will not improve on what you already have
            break
        '''
    if best_y is None:
        best_y = (model.bos_index,)
    return torch.tensor(best_y[1:]), gamma


def _sqd(model, encoder_output, masks, max_output_length,
         labels: dict = None, gamma=float("-inf"), buffer_size=1):
    """
    Lazily generate hypotheses from the model, with options that control the
    order in which hypotheses are generated (currently: depth-first, sorted)
    """
    device = encoder_output.device
    transformer = model.is_transformer
    sparse = model.decoder.gen_func.__name__ != "softmax"

    bos_state = 0.0, SearchState((model.bos_index,), None, None)
    hyps = PQueue()
    hyps.append(bos_state)
    while hyps:
        # pop up to buffer_size hypotheses from the heap
        current_hyps = []
        current_ps = []
        current_dh = []
        current_av = []
        for i in range(buffer_size):
            try:
                p, h = hyps.pop()
                if current_hyps and len(h.y) != len(current_hyps[-1].y):
                    # make sure all hypotheses are the same length
                    hyps.append((p, h))
                    break
                current_hyps.append(h)
                current_ps.append(p)
            except IndexError:
                break
        if current_hyps[0].decoder_hidden is not None:
            if isinstance(current_hyps[0].decoder_hidden, tuple):
                h1, h2 = zip(*[h.decoder_hidden for h in current_hyps])
                current_dh = torch.cat(h1, dim=1), torch.cat(h2, dim=1)
            else:
                current_dh = torch.cat(
                    [h.decoder_hidden for h in current_hyps], dim=1
                )
        else:
            current_dh = None

        if current_hyps[0].att_vectors is not None:
            current_av = torch.stack([h.att_vectors for h in current_hyps])
        else:
            current_av = None

        # tensorize popped scores and hypotheses (this requires them to be the
        # same length, at least for transformers)
        current_p = torch.tensor(current_ps, device=device)
        hyps_tensor = torch.tensor([h.y for h in current_hyps], device=device)
        prev_y = hyps_tensor if transformer else hyps_tensor[:, -1].view(-1, 1)
        assert current_p.size(0) == prev_y.size(0)

        # expand current hyps (i.e. compute their successors) and score them
        # now you need to expand the masks: they need to be matched
        bsz = prev_y.size(0)
        curr_masks = dict()
        for k, v in masks.items():
            curr_masks[k] = v.expand(bsz, v.size(1), v.size(2))
        # ok, also need to expand the encoder output
        new_eo = type(encoder_output)(encoder_output.states, encoder_output.hidden)
        new_eo.tile(bsz, dim=0)
        log_p, decoder_hidden, _, att_vectors = model.decode(
                trg_input=prev_y,
                encoder_output=new_eo,
                masks=curr_masks,
                decoder_hidden=current_dh,
                prev_att_vector=current_av,
                unroll_steps=1,
                labels=labels,
                generate="log"
        )
        p_prime = log_p + current_p.unsqueeze(1)  # beam x |V|

        eos_probs = p_prime[:, model.eos_index]
        top_prob, top_ix = eos_probs.max(dim=0)
        if top_prob > gamma:
            gamma = top_prob
            finished_seq = torch.cat(
                [hyps_tensor[top_ix, 1:], top_ix.new_tensor([model.eos_index])])
            yield top_prob, finished_seq

        # now, take all unfinished hyps that are better than gamma. Add them to
        # the heap.
        true_length = hyps_tensor.size(-1) - 1  # without bos
        if true_length < max_output_length:
            # assumption in this if-clause is that all hypotheses in this beam
            # are the same length
            not_pruned = p_prime.gt(gamma)  # beam x |V|
            not_pruned[:, model.eos_index] = 0  # not sure if this line matters
            still_alive = not_pruned.nonzero()
            # do I want to iterate through the nnz coordinates like this?
            for successor in still_alive:
                hist_i, next_i = successor
                successor_p = p_prime[hist_i, next_i]
                history = current_hyps[hist_i]
                if decoder_hidden is not None:
                    if transformer:
                        # not certain to me that this is correct
                        dh = decoder_hidden[hist_i].unsqueeze(0)
                    else:
                        if isinstance(decoder_hidden, tuple):
                            dh = tuple(dh_i[:, hist_i].unsqueeze(1)
                                       for dh_i in decoder_hidden)
                        else:
                            # print(decoder_hidden.size(), hist_i)
                            dh = decoder_hidden[:, hist_i].unsqueeze(1)
                else:
                    dh = None
                av = att_vectors[hist_i] if att_vectors is not None else None
                new_state = successor_p.item(), SearchState(history.y + (next_i,), dh, av)
                hyps.append(new_state)


def single_queue_decode(
        model, encoder_output, masks, max_output_length, max_hyps=1,
        labels: dict = None, buffer_size: int = 1, gamma=float("-inf")):

    enc_outs = encoder_output.split()
    ys = []
    gammas = []
    for i, enc_out in enumerate(enc_outs):
        seq_masks = {k: v[i].unsqueeze(0) for k, v in masks.items()}
        if model.is_transformer:
            seq_masks["trg"] = torch.ones(1, 1, 1, dtype=torch.bool, device=masks["trg"].device)
        #print(seq_masks["trg"].size())
        if labels is not None:
            seq_labels = {k: v[i].unsqueeze(0) for k, v in labels.items()}

        # as in full_traversal, we have a generator that lazily produces
        # hypotheses
        hypotheses = _sqd(model, enc_out, seq_masks, max_output_length,
                          labels=seq_labels, buffer_size=buffer_size,
                          gamma=gamma)
        total_p = 0.0
        current_best = float("-inf")
        generated = []
        for j in range(max_hyps):
            try:
                p, y = next(hypotheses)
            except StopIteration:
                break
            # print(p, [model.trg_vocab.itos[y_i] for y_i in y])
            generated.append((p, y))
            total_p += math.exp(p)
            if p > current_best:
                current_best = p
        # print()

        best_p, best_y = max(generated) if generated else (float("-inf"), (model.bos_index,))
        ys.append(best_y)
        gammas.append(best_p)

    outputs = pad_and_stack_hyps(ys, model.pad_index)
    # any_finished = outputs.eq(3).any(dim=-1)

    scores = torch.FloatTensor(gammas, device=outputs.device)
    # assert scores[~any_finished].eq(float("-inf")).all()

    return outputs, scores
