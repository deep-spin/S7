from torch.nn import GRU, LSTM
import torch

from joeynmt.model import EncoderOutput
from joeynmt.decoders import RecurrentDecoder
from joeynmt.encoders import RecurrentEncoder
from .test_helpers import TensorTestCase


class TestRecurrentDecoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 10
        self.num_layers = 3
        self.hidden_size = 6
        self.encoder_hidden_size = 3
        self.vocab_size = 5
        seed = 42
        torch.manual_seed(seed)

        bidi_encoder = RecurrentEncoder(emb_size=self.emb_size,
                                        num_layers=self.num_layers,
                                        hidden_size=self.encoder_hidden_size,
                                        bidirectional=True)
        uni_encoder = RecurrentEncoder(emb_size=self.emb_size,
                                       num_layers=self.num_layers,
                                       hidden_size=self.encoder_hidden_size*2,
                                       bidirectional=False)
        self.encoders = [uni_encoder, bidi_encoder]

    def test_recurrent_decoder_size(self):
        # test all combinations of bridge, input_feeding, encoder directions
        for encoder in self.encoders:
            for init_hidden in ["bridge", "zero", "last"]:
                for input_feeding in [True, False]:
                    decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                               encoder_output_size=encoder.output_size,
                                               attention="bahdanau",
                                               emb_size=self.emb_size,
                                               vocab_size=self.vocab_size,
                                               num_layers=self.num_layers,
                                               init_hidden=init_hidden,
                                               input_feeding=input_feeding)
                    self.assertEqual(decoder.rnn.hidden_size, self.hidden_size)
                    self.assertEqual(decoder.att_vector_layer.out_features,
                                     self.hidden_size)
                    self.assertEqual(decoder.output_layer.out_features,
                                     self.vocab_size)
                    self.assertEqual(decoder.output_size, self.vocab_size)
                    self.assertEqual(decoder.rnn.bidirectional, False)

                    self.assertTrue(hasattr(decoder, "bridge_layer"))
                    if init_hidden == "bridge":
                        self.assertEqual(decoder.bridge_layer.mlp[0].out_features,
                                         self.hidden_size)
                        self.assertEqual(decoder.bridge_layer.mlp[0].in_features,
                                         encoder.output_size)
                    elif init_hidden == "last":
                        self.assertTrue(decoder.bridge_layer.mlp is None)
                    else:
                        self.assertTrue(decoder.bridge_layer is None)

                    if input_feeding:
                        self.assertEqual(decoder.rnn_input_size,
                                         self.emb_size + self.hidden_size)
                    else:
                        self.assertEqual(decoder.rnn_input_size, self.emb_size)

    def test_recurrent_decoder_type(self):
        valid_rnn_types = {"gru": GRU, "lstm": LSTM}
        for name, obj in valid_rnn_types.items():
            decoder = RecurrentDecoder(rnn_type=name,
                                       hidden_size=self.hidden_size,
                                       encoder_output_size=self.encoders[0].output_size,
                                       attention="bahdanau",
                                       emb_size=self.emb_size,
                                       vocab_size=self.vocab_size,
                                       num_layers=self.num_layers,
                                       init_hidden="zero",
                                       input_feeding=False)
            self.assertEqual(type(decoder.rnn), obj)

    def test_recurrent_input_dropout(self):
        drop_prob = 0.5
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder_output_size=self.encoders[0].output_size,
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="zero",
                                   input_feeding=False,
                                   dropout=drop_prob,
                                   emb_dropout=drop_prob)
        input_tensor = torch.Tensor([2, 3, 1, -1])
        decoder.train()
        dropped = decoder.emb_dropout(input=input_tensor)
        # eval switches off dropout
        decoder.eval()
        no_drop = decoder.emb_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop - (drop_prob*dropped)).abs().sum(), 0)

        drop_prob = 1.0
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder_output_size=self.encoders[0].output_size,
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="zero",
                                   input_feeding=False,
                                   dropout=drop_prob,
                                   emb_dropout=drop_prob)
        all_dropped = decoder.emb_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        decoder.eval()
        none_dropped = decoder.emb_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_hidden_dropout(self):
        hidden_drop_prob = 0.5
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder_output_size=self.encoders[0].output_size,
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="zero",
                                   input_feeding=False,
                                   hidden_dropout=hidden_drop_prob)
        input_tensor = torch.Tensor([2, 3, 1, -1])
        decoder.train()
        dropped = decoder.hidden_dropout(input=input_tensor)
        # eval switches off dropout
        decoder.eval()
        no_drop = decoder.hidden_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop -
                                 (hidden_drop_prob * dropped)).abs().sum(), 0)

        hidden_drop_prob = 1.0
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder_output_size=self.encoders[0].output_size,
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="zero",
                                   input_feeding=False,
                                   hidden_dropout=hidden_drop_prob)
        all_dropped = decoder.hidden_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        decoder.eval()
        none_dropped = decoder.hidden_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_freeze(self):
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder_output_size=self.encoders[0].output_size,
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="zero",
                                   input_feeding=False,
                                   freeze=True)
        for n, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_recurrent_forward(self):
        time_dim = 4
        batch_size = 2
        # make sure the outputs match the targets
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder_output_size=self.encoders[0].output_size,
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="zero",
                                   input_feeding=False)
        encoder_states = torch.rand(size=(batch_size, time_dim,
                                          self.encoders[0].output_size))
        trg_inputs = torch.ones(size=(batch_size, time_dim, self.emb_size))
        # no padding, no mask
        #x_length = torch.Tensor([time_dim]*batch_size).int()
        mask = torch.ones(size=(batch_size, 1, time_dim)).byte()
        encoder_output = EncoderOutput(
            encoder_states, encoder_states[:, -1, :]
        )
        output, hidden, att_probs, att_vectors = decoder(
            trg_inputs, #encoder_hidden=encoder_states[:, -1, :],
            encoder_output=encoder_output, masks=mask, unroll_steps=time_dim,
            hidden=None, prev_att_vector=None)
        att_probs = att_probs["src_trg"]
        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.vocab_size]))
        self.assertEqual(hidden.shape, torch.Size(
            [self.num_layers, batch_size, self.hidden_size]))
        self.assertEqual(att_probs.shape, torch.Size(
            [batch_size, time_dim, time_dim]))
        self.assertEqual(att_vectors.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        hidden_target = torch.Tensor(
            [[[-0.4330,  0.0563, -0.3310,  0.4228, -0.1188, -0.0436],
            [-0.4330,  0.0563, -0.3310,  0.4228, -0.1188, -0.0436]],

            [[ 0.1796, -0.0573,  0.3581, -0.0051, -0.3506,  0.2007],
            [ 0.1796, -0.0573,  0.3581, -0.0051, -0.3506,  0.2007]],

            [[-0.1954, -0.2804, -0.1885, -0.2336, -0.4033,  0.0890],
            [-0.1954, -0.2804, -0.1885, -0.2336, -0.4033,  0.0890]]]
        )
        output_target = torch.Tensor(
            [[[-0.1533,  0.1284, -0.1100, -0.0350, -0.1126],
            [-0.1260,  0.1000, -0.1006, -0.0328, -0.0942],
            [-0.1052,  0.0845, -0.0984, -0.0327, -0.0839],
            [-0.0899,  0.0753, -0.0986, -0.0330, -0.0779]],

            [[-0.1302,  0.1310, -0.0881, -0.0362, -0.1239],
            [-0.1026,  0.1024, -0.0786, -0.0340, -0.1054],
            [-0.0817,  0.0867, -0.0765, -0.0339, -0.0951],
            [-0.0663,  0.0775, -0.0766, -0.0343, -0.0890]]]
        )
        att_vectors_target = torch.Tensor(
            [[[-0.0351,  0.1532,  0.0301, -0.1575,  0.0526, -0.2428],
            [-0.0727,  0.1208,  0.0664, -0.1267,  0.0610, -0.2101],
            [-0.0964,  0.0932,  0.0850, -0.1058,  0.0717, -0.1949],
            [-0.1115,  0.0725,  0.0942, -0.0914,  0.0810, -0.1871]],

            [[ 0.0667,  0.1424, -0.1167, -0.1500, -0.0087, -0.2175],
            [ 0.0290,  0.1099, -0.0807, -0.1191, -0.0004, -0.1845],
            [ 0.0052,  0.0821, -0.0619, -0.0981,  0.0103, -0.1691],
            [-0.0101,  0.0614, -0.0527, -0.0836,  0.0195, -0.1613]]]
        )
        self.assertTensorAlmostEqual(hidden_target, hidden)
        self.assertTensorAlmostEqual(output_target, output)
        self.assertTensorAlmostEqual(att_vectors, att_vectors_target)
        # att_probs should be a distribution over the output vocabulary
        self.assertTensorAlmostEqual(att_probs.sum(2),
                                     torch.ones(batch_size, time_dim))
