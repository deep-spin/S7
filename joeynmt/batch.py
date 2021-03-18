# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        # Will the ordering of fields be consistent across batches?
        # Does it matter?
        self.fields = tuple(torch_batch.fields)  # is this used externally?
        self._data = dict()
        self._labels = set()
        for f_name in torch_batch.fields:
            torch_field = getattr(torch_batch, f_name)
            if isinstance(torch_field, tuple):
                f, f_lengths = torch_field
                self.nseqs = f.size(0)
                if f_name != "trg":
                    f_mask = (f != pad_index).unsqueeze(1)
                else:
                    f_mask = (f[:, :-1] != pad_index).unsqueeze(1)
                field_data = f, f_lengths, f_mask
            else:
                assert isinstance(torch_field, torch.Tensor)
                field_data = torch_field
                self._labels.add(f_name)
            self._data[f_name] = field_data

        if "trg" in self._data:
            trg = self._data["trg"][0]
            self.ntokens = (trg != pad_index).data.sum().item()

    def items(self, include_trg=False):
        for item in self._data.items():
            name = item[0]
            if name not in self._labels and (include_trg or name != "trg"):
                yield item

    @property
    def labels(self):
        return {k: v for k, v in self._data.items() if k in self._labels}

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    @property
    def masks(self):
        return {f_name: tensors[2] for f_name, tensors in self._data.items()
                if f_name not in self._labels}

    @property
    def lengths(self):
        return {f_name: tensors[1] for f_name, tensors in self._data.items()
                if f_name not in self._labels}

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        # note that it will use src_len if it is present, regardless of
        # other src fields
        if "src" in self._data:
            _, perm_ix = self["src"][1].sort(0, descending=True)
        else:
            lengths = [v for k, v in self.lengths.items() if k != "trg"]
            total_lengths = torch.stack(lengths).sum(dim=0)
            _, perm_ix = total_lengths.sort(0, descending=True)
        rev_index = [0] * perm_ix.size(0)
        for new_pos, old_pos in enumerate(perm_ix):
            rev_index[old_pos] = new_pos

        for f_name, f_values in self._data.items():
            if f_name in self._labels:
                self._data[f_name] = f_values[perm_ix]
            else:
                seq, leng, mask = f_values
                new_values = seq[perm_ix], leng[perm_ix], mask[perm_ix]
                self._data[f_name] = new_values

        return rev_index
