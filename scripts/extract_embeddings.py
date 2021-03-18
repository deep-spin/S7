#!/usr/bin/env python

import argparse
from os.path import join, basename
from os import makedirs
from glob import glob

import torch


def read_vocab(path):
    with open(path) as f:
        return [line.strip() for line in f]


def embedding_name(model_key):
    """
    'src_embed.query_emb.language.lut.weight',
    'src_embed.query_emb.main.lut.weight',
    'trg_embed.lut.weight'"""
    if "embed" in model_key and model_key.endswith("lut.weight"):
        name_parts = model_key.split('.')
        if name_parts[0] == "embeds":
            field_name = name_parts[1]
        else:
            field_name = name_parts[0].split("_")[0]
        return field_name
    else:
        return None


def main(opt):
    ckpt = torch.load(join(opt.model_path, "best.ckpt"), map_location="cpu")
    vocabs = {basename(path).split("_")[0]: read_vocab(path)
              for path in glob(join(opt.model_path, "*vocab.txt"))}

    model_state = ckpt["model_state"]
    embeddings = {embedding_name(k): v
                  for k, v in model_state.items() if embedding_name(k)}
    output_dir = join(opt.model_path, "embeddings")
    makedirs(output_dir, exist_ok=True)
    for field in embeddings:
        vocab = vocabs[field]
        emb_matrix = embeddings[field]
        assert len(vocab) == emb_matrix.size(0)
        with open(join(output_dir, field + ".tsv"), "w") as f:
            for word_type, word_emb in zip(vocab, emb_matrix):
                vec_str = [str(i) for i in word_emb.tolist()]
                toks = [word_type] + vec_str
                f.write("\t".join(toks) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    opt = parser.parse_args()
    main(opt)
