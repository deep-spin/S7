import math
import torch
from torch import nn, Tensor
from joeynmt.helpers import freeze_params


class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)


class FeatureEmbeddings(nn.ModuleDict):
    """
    An embedding module, but with added side information (for example,
    a language embedding).
    """
    def __init__(self, main_emb, mode='token', **kwargs):
        assert mode in ['token', 'feature', None]
        super(FeatureEmbeddings, self).__init__(
            {k: v for k, v in kwargs.items() if v is not None}
        )
        self._order = sorted(k for k in self.keys())
        self['main'] = main_emb

        if mode == 'feature':
            self.embedding_dim = sum(m.embedding_dim for m in self.values())
        else:
            self.embedding_dim = main_emb.embedding_dim
            assert all(m.embedding_dim == self.embedding_dim
                       for m in self.values())

        self.mode = mode

    def forward(self, input, **kwargs):
        """
        input (LongTensor): sequence length x batch size
        additional arguments:
        """
        main_emb = self['main'](input)
        if self.mode is None:
            return main_emb
        seq_len = main_emb.size(1)
        embs = []
        for k in self._order:
            side_info = kwargs[k]
            side_emb = self[k](side_info)

            if side_emb.dim() == 2:  # is this always true?
                side_emb = side_emb.unsqueeze(1)
                if self.mode == 'feature':
                    side_emb = side_emb.expand(-1, seq_len, -1)
            embs.append(side_emb)
        embs.append(main_emb)
        cat_dim = 2 if self.mode == 'feature' else 1
        emb = torch.cat(embs, cat_dim)
        return emb

    @property
    def num_embeddings(self):
        """Vocab size for the main embedding level"""
        return self["main"].num_embeddings
