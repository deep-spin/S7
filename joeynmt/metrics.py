# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from functools import partial
import sacrebleu
import numpy as np

from joeynmt.helpers import postprocess


def chrf(hypotheses, references):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses,
                                 references=references).score


def bleu(hypotheses, references):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.raw_corpus_bleu(sys_stream=hypotheses,
                                     ref_streams=[references]).score


def token_accuracy(hypotheses, references, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    tok_fun = partial(str.split, sep=" ") if level in ["word", "bpe"] else list
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(tok_fun(hyp), tok_fun(ref)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_seqs = sum(hyp == ref for hyp, ref in zip(hypotheses, references))
    return (correct_seqs / len(hypotheses)) * 100 if hypotheses else 0.0


def word_error_rate(hypotheses, references):
    return 100 - sequence_accuracy(hypotheses, references)


def _levenshtein(str1, str2):
    m = np.zeros((len(str2)+1, len(str1)+1), dtype=int)
    m[:, 0] = np.arange(len(str2) + 1, dtype=int)
    m[0] = np.arange(len(str1) + 1, dtype=int)
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            dg = int(str1[y-1] != str2[x-1])
            m[x, y] = min(m[x-1, y] + 1, m[x, y-1] + 1, m[x-1, y-1] + dg)
    return m[len(str2), len(str1)]


def character_error_rate(hypotheses, references, level="word"):
    tok_fun = partial(str.split, sep=" ") if level in ["word", "bpe"] else list
    assert len(hypotheses) == len(references)
    total_length = 0
    total_distance = 0
    for hypothesis, reference in zip(hypotheses, references):
        h = tok_fun(hypothesis)
        r = tok_fun(reference)
        total_distance += _levenshtein(h, r)
        total_length += len(r)
    return 100 * total_distance / total_length


def levenshtein_distance(hypotheses, references, level="word"):
    """
    Average levenshtein distance: this differs from
    """
    tok_fun = partial(str.split, sep=" ") if level in ["word", "bpe"] else list
    assert len(hypotheses) == len(references)
    total_distance = 0
    for hypothesis, reference in zip(hypotheses, references):
        h = tok_fun(hypothesis)
        r = tok_fun(reference)
        total_distance += _levenshtein(h, r)
    return total_distance / len(references)


def levenshtein_distance_cost(hypothesis, reference, **kwargs):
    return float(_levenshtein(hypothesis, reference))


def sentence_bleu_cost(hypothesis, reference, level="word"):
    # join the tokens
    hypothesis = postprocess([hypothesis], level=level)[0]
    reference = postprocess([reference], level=level)[0]
    return 100 - sacrebleu.sentence_bleu(hypothesis, reference, smooth_value=1.0).score
