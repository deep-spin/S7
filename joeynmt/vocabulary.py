# coding: utf-8

"""
Vocabulary module
"""
from itertools import chain
from collections import defaultdict, Counter
from typing import List
import numpy as np

from joeynmt.constants import UNK_TOKEN, DEFAULT_UNK_ID, \
    EOS_TOKEN, BOS_TOKEN, PAD_TOKEN


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str], type_frequencies=None) -> None:
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size

        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        self.stoi = defaultdict(lambda: DEFAULT_UNK_ID)
        self.itos = []
        if tokens is not None:
            for t in chain(self.specials, tokens):
                # add to vocab if not already there
                if t not in self.stoi:
                    self.stoi[t] = len(self.itos)
                    self.itos.append(t)
            assert len(self.stoi) == len(self.itos)

        if type_frequencies is not None:
            self.frequencies = [type_frequencies[s] for s in self.stoi]
        else:
            self.frequencies = None

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r") as open_file:
            tokens = [line.strip("\n") for line in open_file]
        return cls(tokens)

    @classmethod
    def from_dataset(cls, dataset, fields, max_size: int, min_freq: int,
                     sequential: bool = True):
        if isinstance(fields, str):
            fields = [fields]
        if sequential:
            tokens = chain.from_iterable(
                [getattr(ex, field)
                 for field in fields
                 for ex in dataset.examples]
            )
        else:
            tokens = [getattr(ex, field)
                      for field in fields
                      for ex in dataset.examples]
        type_frequencies = Counter(tokens)

        if min_freq > 1:
            type_frequencies = filter_min(type_frequencies, min_freq)
        vocab_types = sort_and_cut(type_frequencies, max_size)
        assert len(vocab_types) <= max_size

        # I think this will be wrong if sort_and_cut did anything
        type_frequencies[EOS_TOKEN] = len(dataset.examples)

        return cls(vocab_types, type_frequencies=type_frequencies)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, f: str) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param f: path to file where the vocabulary is written
        """
        with open(f, "w") as open_file:
            open_file.write("\n".join(self.itos) + "\n")

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == DEFAULT_UNK_ID

    def __len__(self) -> int:
        return len(self.itos)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) \
            -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


def filter_min(counter: Counter, min_freq: int):
    """ Filter counter by min frequency """
    return Counter({t: c for t, c in counter.items() if c >= min_freq})


def sort_and_cut(counter: Counter, limit: int):
    """ Cut counter to most frequent,
    sorted numerically and alphabetically"""
    # ignoring the alphabetical part, it's fine to do
    # [word_type for (word_type, count) in counter.most_common(limit)]
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens
