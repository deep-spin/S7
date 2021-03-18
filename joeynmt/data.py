# coding: utf-8
"""
Data module
"""
import sys
from os.path import isfile
from functools import partial
import unicodedata

import torch

from torchtext.datasets import TranslationDataset
from torchtext.data import Dataset, Iterator, Field, BucketIterator

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.datasets import MonoDataset, TSVDataset


def filter_example(ex, max_sent_length, keys):
    return all(len(getattr(ex, k)) <= max_sent_length for k in keys)


def decompose_tokenize(string):
    string = unicodedata.normalize("NFD", string)
    return list(string)


def load_data(data_cfg: dict):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - vocabs: dictionary from src and trg (and possibly other fields) to
            their corresponding vocab objects
    """
    data_format = data_cfg.get("format", "bitext")
    formats = {"bitext", "tsv"}
    assert data_format in formats

    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)

    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]
    default_level = data_cfg.get("level", "word")
    voc_limit = data_cfg.get("voc_limit", sys.maxsize)
    voc_min_freq = data_cfg.get("voc_min_freq", 1)

    tokenizers = {
        "word": str.split,
        "bpe": str.split,
        "char": list,
        "char-decompose": decompose_tokenize,
        "tag": partial(str.split, sep=";")
    }

    # is field_names better?
    # column_labels seems like an ok name for the
    all_fields = data_cfg.get("columns", ["src", "trg"])
    label_fields = data_cfg.get("label_fields", [])
    assert all(label in all_fields for label in label_fields)
    sequential_fields = [field for field in all_fields
                         if field not in label_fields]
    src_fields = [f_name for f_name in all_fields if f_name != "trg"]
    trg_fields = ["trg"]

    suffixes = {f_name: data_cfg.get(f_name, "")
                for f_name in sequential_fields}

    seq_field_cls = partial(
        Field,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=lowercase,
        include_lengths=True
    )

    fields = dict()
    # what are the source fields? what are the target fields?

    for f_name in sequential_fields:
        bos = BOS_TOKEN if f_name == "trg" else None
        current_level = data_cfg.get(f_name + "_level", default_level)
        assert current_level in tokenizers, "Invalid tokenization level"
        tok_fun = tokenizers[current_level]

        fields[f_name] = seq_field_cls(init_token=bos, tokenize=tok_fun)

    for f_name in label_fields:
        fields[f_name] = Field(sequential=False)

    filter_ex = partial(
        filter_example,
        max_sent_length=max_sent_length,
        keys=tuple(fields.keys())
    )

    if data_format == "bitext":
        dataset_cls = partial(
            TranslationDataset,
            exts=("." + suffixes["src"], "." + suffixes["trg"]),
            fields=(fields["src"], fields["trg"])
        )
    else:
        dataset_cls = partial(TSVDataset, fields=fields, columns=all_fields,
                              label_columns=label_fields)

    if test_path is not None:
        trg_suffix = suffixes["trg"]
        if data_format != "bitext" or isfile(test_path + "." + trg_suffix):
            test_dataset_cls = dataset_cls
        else:
            test_dataset_cls = partial(
                MonoDataset, ext="." + suffixes["src"], field=fields["src"]
            )
    else:
        test_dataset_cls = None

    train_data = dataset_cls(path=train_path, filter_pred=filter_ex)

    vocabs = dict()

    # here's the thing: you want to have a vocab for each f_name, but not
    # necessarily a *distinct* vocab
    share_src_vocabs = data_cfg.get("share_src_vocabs", False)
    if share_src_vocabs:
        field_groups = [src_fields, trg_fields]
    else:
        field_groups = [[f] for f in sequential_fields]
    for f_group in field_groups:
        if len(f_group) == 1:
            f_name = f_group[0]
            max_size = data_cfg.get("{}_voc_limit".format(f_name), voc_limit)
            min_freq = data_cfg.get(
                "{}_voc_min_freq".format(f_name), voc_min_freq
            )
            vocab_file = data_cfg.get("{}_vocab".format(f_name), None)
        else:
            # multiple fields sharing a vocabulary
            max_size = voc_limit
            min_freq = voc_min_freq
            vocab_file = None

        if vocab_file is not None:
            f_vocab = Vocabulary.from_file(vocab_file)
        else:
            f_vocab = Vocabulary.from_dataset(
                train_data, f_group, max_size, min_freq
            )
        for f_name in f_group:
            vocabs[f_name] = f_vocab

    label_field_groups = [[lf] for lf in label_fields]
    for f_group in label_field_groups:
        if len(f_group) == 1:
            f_name = f_group[0]
            max_size = data_cfg.get("{}_voc_limit".format(f_name), voc_limit)
            min_freq = data_cfg.get(
                "{}_voc_min_freq".format(f_name), voc_min_freq
            )
            vocab_file = data_cfg.get("{}_vocab".format(f_name), None)
        else:
            # multiple fields sharing a vocabulary
            max_size = voc_limit
            min_freq = voc_min_freq
            vocab_file = None

        if vocab_file is not None:
            f_vocab = Vocabulary.from_file(vocab_file)
        else:
            f_vocab = Vocabulary.from_dataset(
                train_data, f_group, max_size, min_freq, sequential=False
            )
        for f_name in f_group:
            vocabs[f_name] = f_vocab

    '''
    for vocab_name, vocab in vocabs.items():
        print(vocab_name)
        print(vocab.itos)
        print()
    '''

    dev_data = dataset_cls(path=dev_path)

    if test_path is not None:
        trg_suffix = suffixes["trg"]
        if data_format != "bitext" or isfile(test_path + "." + trg_suffix):
            test_dataset_cls = dataset_cls
        else:
            test_dataset_cls = partial(
                MonoDataset, ext="." + suffixes["src"], field=fields["src"]
            )
        test_data = test_dataset_cls(path=test_path)
    else:
        test_data = None

    for field_name in fields:
        fields[field_name].vocab = vocabs[field_name]

    ret = {"train_data": train_data, "dev_data": dev_data,
           "test_data": test_data, "vocabs": vocabs}
    return ret


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def src_len(ex):
    return len(ex.src)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   valid_teacher_forcing: bool = False,
                   shuffle: bool = False,
                   use_cuda: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    sort_key = src_len if isinstance(dataset, TranslationDataset) else None

    if train:
        # optionally shuffle and sort during training
        data_iter = BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=sort_key,
            shuffle=shuffle,
            device=device)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = BucketIterator(
            repeat=False, sort=valid_teacher_forcing, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            sort_within_batch=valid_teacher_forcing,
            train=False, device=device)

    return data_iter
