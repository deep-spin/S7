import torch

from joeynmt.model import EncoderOutput
from joeynmt.decoders import TransformerDecoder, TransformerDecoderLayer
from .test_helpers import TensorTestCase


class TestTransformerDecoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.
        self.seed = 42

    def test_transformer_decoder_freeze(self):
        torch.manual_seed(self.seed)
        encoder = TransformerDecoder(freeze=True)
        for n, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_decoder_output_size(self):

        vocab_size = 11
        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, vocab_size=vocab_size)

        if not hasattr(decoder, "output_size"):
            self.fail("Missing output_size property.")

        self.assertEqual(decoder.output_size, vocab_size)

    def test_transformer_decoder_forward(self):
        torch.manual_seed(self.seed)
        batch_size = 2
        src_time_dim = 4
        trg_time_dim = 5
        vocab_size = 7

        trg_embed = torch.rand(size=(batch_size, trg_time_dim, self.emb_size))

        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, emb_dropout=self.dropout,
            vocab_size=vocab_size)

        encoder_output = EncoderOutput(
            torch.rand(size=(batch_size, src_time_dim, self.hidden_size)),
            None)

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)) == 1

        output, states, _, _ = decoder(
            trg_embed, encoder_output, src_mask, trg_mask)

        output_target = torch.Tensor(
            [[[ 0.1765,  0.4578,  0.2345, -0.5303,  0.3862,  0.0964,  0.6882],
            [ 0.3363,  0.3907,  0.2210, -0.5414,  0.3770,  0.0748,  0.7344],
            [ 0.3275,  0.3729,  0.2797, -0.3519,  0.3341,  0.1605,  0.5403],
            [ 0.3081,  0.4513,  0.1900, -0.3443,  0.3072,  0.0570,  0.6652],
            [ 0.3253,  0.4315,  0.1227, -0.3371,  0.3339,  0.1129,  0.6331]],

            [[ 0.3235,  0.4836,  0.2337, -0.4019,  0.2831, -0.0260,  0.7013],
            [ 0.2800,  0.5662,  0.0469, -0.4156,  0.4246, -0.1121,  0.8110],
            [ 0.2968,  0.4777,  0.0652, -0.2706,  0.3146,  0.0732,  0.5362],
            [ 0.3108,  0.4910,  0.0774, -0.2341,  0.2873,  0.0404,  0.5909],
            [ 0.2338,  0.4371,  0.1350, -0.1292,  0.0673,  0.1034,  0.5356]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[ 8.3742e-01, -1.3161e-01,  2.1876e-01, -1.3920e-01, -9.1572e-01,
            2.3006e-01,  3.8328e-01, -1.6271e-01,  3.7370e-01, -1.2110e-01,
            -4.7549e-01, -4.0622e-01],
            [ 8.3609e-01, -2.9161e-02,  2.0583e-01, -1.3571e-01, -8.0510e-01,
            2.7630e-01,  4.8219e-01, -1.8863e-01,  1.1977e-01, -2.0179e-01,
            -4.4314e-01, -4.1228e-01],
            [ 8.5478e-01,  1.1368e-01,  2.0400e-01, -1.3059e-01, -8.1042e-01,
            1.6369e-01,  5.4244e-01, -2.9103e-01,  3.9919e-01, -3.3826e-01,
            -4.5423e-01, -4.2516e-01],
            [ 9.0388e-01,  1.1853e-01,  1.9927e-01, -1.1675e-01, -7.7208e-01,
            2.0686e-01,  4.6024e-01, -9.1610e-02,  3.9778e-01, -2.6214e-01,
            -4.7688e-01, -4.0807e-01],
            [ 8.9476e-01,  1.3646e-01,  2.0298e-01, -1.0910e-01, -8.2137e-01,
            2.8025e-01,  4.2538e-01, -1.1852e-01,  4.1497e-01, -3.7422e-01,
            -4.9212e-01, -3.9790e-01]],

            [[ 8.8745e-01, -2.5798e-02,  2.1483e-01, -1.8219e-01, -6.4821e-01,
            2.6279e-01,  3.9598e-01, -1.0423e-01,  3.0726e-01, -1.1315e-01,
            -4.7201e-01, -3.6979e-01],
            [ 7.5528e-01,  6.8919e-02,  2.2486e-01, -1.6395e-01, -7.9692e-01,
            3.7830e-01,  4.9367e-01,  2.4355e-02,  2.6674e-01, -1.1740e-01,
            -4.4945e-01, -3.6367e-01],
            [ 8.3467e-01,  1.7779e-01,  1.9504e-01, -1.6034e-01, -8.2783e-01,
            3.2627e-01,  5.0045e-01, -1.0181e-01,  4.4797e-01, -4.8046e-01,
            -3.7264e-01, -3.7392e-01],
            [ 8.4359e-01,  2.2699e-01,  1.9721e-01, -1.5768e-01, -7.5897e-01,
            3.3738e-01,  4.5559e-01, -1.0258e-01,  4.5782e-01, -3.8058e-01,
            -3.9275e-01, -3.8412e-01],
            [ 9.6349e-01,  1.6264e-01,  1.8207e-01, -1.6910e-01, -5.9304e-01,
            1.4468e-01,  2.4968e-01,  6.4794e-04,  5.4930e-01, -3.8420e-01,
            -4.2137e-01, -3.8016e-01]]]
        )

        self.assertEqual(states_target.shape, states.shape)
        self.assertTensorAlmostEqual(states_target, states)

    def test_transformer_decoder_layers(self):

        torch.manual_seed(self.seed)
        batch_size = 2
        src_time_dim = 4
        trg_time_dim = 5
        vocab_size = 7

        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, vocab_size=vocab_size)

        self.assertEqual(len(decoder.layers), self.num_layers)

        for layer in decoder.layers:
            self.assertTrue(isinstance(layer, TransformerDecoderLayer))
            self.assertTrue(hasattr(layer, "src_trg_att"))
            self.assertTrue(hasattr(layer, "trg_trg_att"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].in_features, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].out_features, self.ff_size)
