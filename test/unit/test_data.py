import unittest

import numpy as np

from joeynmt.data import MonoDataset, TranslationDataset, load_data, \
    make_data_iter


class TestData(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.levels = ["char", "word"]  # bpe is equivalently processed to word
        self.max_sent_length = 10

        # minimal data config
        self.data_cfg = {"src": "de", "trg": "en", "train": self.train_path,
                         "dev": self.dev_path,
                         "max_sent_length": self.max_sent_length}

    def testIteratorBatchType(self):

        current_cfg = self.data_cfg.copy()
        current_cfg["level"] = "word"
        current_cfg["lowercase"] = False

        # load toy data
        data = load_data(current_cfg)
        train_data = data["train_data"]
        dev_data = data["dev_data"]
        test_data = data["test_data"]
        vocabs = data["vocabs"]
        src_vocab = vocabs["src"]
        trg_vocab = vocabs["trg"]

        # make batches by number of sentences
        train_iter = iter(make_data_iter(
            train_data, batch_size=10, batch_type="sentence"))
        batch = next(train_iter)

        self.assertEqual(batch.src[0].shape[0], 10)
        self.assertEqual(batch.trg[0].shape[0], 10)

        # make batches by number of tokens
        train_iter = iter(make_data_iter(
            train_data, batch_size=100, batch_type="token"))
        _ = next(train_iter)  # skip a batch
        _ = next(train_iter)  # skip another batch
        batch = next(train_iter)

        self.assertEqual(batch.src[0].shape[0], 8)
        self.assertEqual(np.prod(batch.src[0].shape), 88)
        self.assertLessEqual(np.prod(batch.src[0].shape), 100)

    def testDataLoading(self):
        # test all combinations of configuration settings
        for test_path in [None, self.test_path]:
            for level in self.levels:
                for lowercase in [True, False]:
                    current_cfg = self.data_cfg.copy()
                    current_cfg["level"] = level
                    current_cfg["lowercase"] = lowercase
                    if test_path is not None:
                        current_cfg["test"] = test_path

                    # load the data
                    data = load_data(current_cfg)
                    train_data = data["train_data"]
                    dev_data = data["dev_data"]
                    test_data = data["test_data"]
                    vocabs = data["vocabs"]
                    src_vocab = vocabs["src"]
                    trg_vocab = vocabs["trg"]

                    self.assertIs(type(train_data), TranslationDataset)
                    self.assertIs(type(dev_data), TranslationDataset)
                    if test_path is not None:
                        # test has no target side
                        self.assertIs(type(test_data), MonoDataset)

                    # check the number of examples loaded
                    # training set is filtered to max_sent_length
                    expected_train_len = 5 if level == "char" else 382
                    expected_testdev_len = 20  # dev and test have the same len
                    self.assertEqual(len(train_data), expected_train_len)
                    self.assertEqual(len(dev_data), expected_testdev_len)
                    if test_path is None:
                        self.assertIsNone(test_data)
                    else:
                        self.assertEqual(len(test_data), expected_testdev_len)

                    # check the segmentation: src and trg attributes are lists
                    for corpus in [train_data, dev_data]:
                        for side in ["src", "trg"]:
                            toks = corpus.examples[0].__dict__[side]
                            self.assertIs(type(toks), list)
                    if test_path is not None:
                        self.assertIs(type(test_data.examples[0].src), list)
                        self.assertFalse(hasattr(test_data.examples[0], "trg"))

                    # check the length filtering of the training examples
                    for side in ["src", "trg"]:
                        self.assertFalse(
                            any(len(ex.__dict__[side]) > self.max_sent_length
                                for ex in train_data.examples)
                        )

                    # check the lowercasing
                    if lowercase:
                        for corpus in [train_data, dev_data]:
                            for side in ["src", "trg"]:
                                self.assertTrue(
                                    all(" ".join(ex.__dict__[side]).islower()
                                        for ex in corpus.examples)
                                )
                        if test_path is not None:
                            self.assertTrue(
                                all(" ".join(ex.src).islower()
                                    for ex in test_data.examples)
                            )

                    # check the first example from the training set
                    expected_srcs = {"char": "Danke.",
                                     "word": "David Gallo: Das ist Bill Lange."
                                             " Ich bin Dave Gallo."}
                    expected_trgs = {"char": "Thank you.",
                                     "word": "David Gallo: This is Bill Lange. "
                                             "I'm Dave Gallo."}
                    exp_src = expected_srcs[level]
                    exp_trg = expected_trgs[level]
                    if lowercase:
                        exp_src = exp_src.lower()
                        exp_trg = exp_trg.lower()
                    if level == "char":
                        comp_src = list(exp_src)
                        comp_trg = list(exp_trg)
                    else:
                        comp_src = exp_src.split()
                        comp_trg = exp_trg.split()
                    self.assertEqual(train_data.examples[0].src, comp_src)
                    self.assertEqual(train_data.examples[0].trg, comp_trg)
