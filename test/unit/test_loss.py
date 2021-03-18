import torch
import torch.nn as nn

from joeynmt.loss import LabelSmoothingLoss
from .test_helpers import TensorTestCase


class TestTransformerUtils(TensorTestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)

    def test_label_smoothing(self):
        pad_index = 0
        smoothing = 0.4
        criterion = LabelSmoothingLoss(ignore_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor(
            [[[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]
        )

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1],
                                    [2, 0],
                                    [1, 0]])

        # test the smoothing function
        smoothed_targets = criterion._smooth_targets(targets=targets.view(-1),
                                      vocab_size=predict.size(-1))
        self.assertTensorAlmostEqual(
            smoothed_targets,
            torch.Tensor(
                [[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                 [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                 [0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        )
        assert torch.max(smoothed_targets) == 1-smoothing

        # test the loss computation
        v = criterion(predict.log(), targets)
        self.assertTensorAlmostEqual(v, 2.1326)
