#!/usr/bin/env python

import argparse
import unicodedata
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from plotnine import ggplot, aes, geom_point


def read_embeddings(path):
    labels, vectors = [], []
    with open(path) as f:
        for line in f:
            toks = line.rstrip().split('\t')
            labels.append(toks[0])
            vectors.append([float(t) for t in toks[1:]])

    return labels, np.array(vectors, dtype=np.float)


def build_tsne(path, scripts=None, **kwargs):
    labels, vectors = read_embeddings(path)
    if scripts is not None:
        label_scripts = [script(label) for label in labels]
        mask = [label_script in scripts for label_script in label_scripts]
        labels = [label for label, m in zip(labels, mask) if m]
        vectors = vectors[mask]
    tsne_vec = TSNE(**kwargs).fit_transform(vectors)
    df = pd.DataFrame(tsne_vec, columns=["x", "y"])
    df["label"] = labels
    df["script"] = df["label"].apply(script)
    return df


def script(s): 
    try: 
        return unicodedata.name(s).split()[0] 
    except TypeError: 
        return "other" 


def main(opts):
    labels, vectors = read_embeddings(opt.embeddings)
    tsne_vec = TSNE().fit_transform(vectors)
    print(tsne_vec.shape)
    df = pd.DataFrame(tsne_vec, columns=["x", "y"])
    df["label"] = labels
    print(df)
    ggplot(aes(x="x", y="y", color="label"), data=df) + geom_point()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    opt = parser.parse_args()
    main(opt)
