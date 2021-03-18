# coding: utf-8

from os.path import expanduser
from glob import glob
from itertools import chain

from torchtext.data import Dataset, Field, Example


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(Example.fromlist([src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)


class TSVDataset(Dataset):

    def sort_key(self, ex):
        longest_src = max(len(getattr(ex, c))
                          for c in self._src_columns)
        if hasattr(ex, "trg"):
            return longest_src, len(ex.trg)
        return longest_src

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(TSVDataset, self).__reduce_ex__()

    def __init__(self, fields, path, filter_pred=None, columns=("src", "trg"),
                 label_columns=()):
        """
        Note that tsv does not currently allow missing columns (such as when
        translating a file with no trg specified)
        """
        self._columns = columns
        self._src_columns = [c for c in columns
                             if c != "trg" and c not in label_columns]
        self.label_columns = label_columns
        fields = {k: [(k, v)] for k, v in fields.items()}
        paths = glob(path) if isinstance(path, str) else path
        assert len(paths) > 0
        paths.sort()
        examples = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ex_dict = dict()
                        values = line.strip().split('\t')
                        assert len(values) == len(columns), \
                            "Wrong number of columns"
                        for column, value in zip(columns, values):
                            ex_dict[column] = value

                        ex = Example.fromdict(ex_dict, fields)
                        examples.append(ex)

        fields = dict(chain.from_iterable(fields.values()))
        super(TSVDataset, self).__init__(examples, fields, filter_pred)
