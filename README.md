# &nbsp; ![Joey-NMT](joey-small.png) possum-nmt: seq2seq modeling for marsupials
[![Build Status](https://travis-ci.com/joeynmt/joeynmt.svg?branch=master)](https://travis-ci.org/joeynmt/joeynmt)


## Goal and Purpose
This is DeepSPIN's spinoff (a DeepSPINoff) of Joey NMT.

## Features
Joey NMT implements the following features (aka the minimalist toolkit of NMT):
- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based input handling
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization
- Learning curve plotting

## Installation
Joey NMT is built on [PyTorch](https://pytorch.org/) and [torchtext](https://github.com/pytorch/text) for Python >= 3.5.

1. Clone this repository:
`git clone https://github.com/joeynmt/joeynmt.git`
2. Install joeynmt and its requirements:
`cd joeynmt`
`pip3 install .` (you might want to add `--user` for a local installation).
3. Run the unit tests:
`python3 -m unittest`

### Data Preparation
TODO: write some up-to-date stuff

#### Parallel Data
For training a translation model, you need parallel data, i.e. a collection of source sentences and reference translations that are aligned sentence-by-sentence and stored in two files, 
such that each line in the reference file is the translation of the same line in the source file.

#### Pre-processing
Before training a model on it, parallel data is most commonly filtered by length ratio, tokenized and true- or lowercased.

The Moses toolkit provides a set of useful [scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts) for this purpose.

In addition, you might want to build the NMT model not on the basis of words, but rather sub-words or characters (the `level` in JoeyNMT configurations).
Currently, JoeyNMT supports the byte-pair-encodings (BPE) format by [subword-nmt](https://github.com/rsennrich/subword-nmt).

### Configuration
Experiments are specified in configuration files, in simple [YAML](http://yaml.org/) format. You can find examples in the `configs` directory.
`small.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN), 
paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

### Training

#### Start
For training, run 

`python3 -m joeynmt train configs/small.yaml`. 

This will train a model on the training data specified in the config (here: `small.yaml`), 
validate on validation data, 
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the `model_dir` (also specified in config).

Note that pre-processing like tokenization or BPE-ing is not included in training, but has to be done manually before.

Tip: Be careful not to overwrite models, set `overwrite: False` in the model configuration.

#### Validations
The `validations.txt` file in the model directory reports the validation results at every validation point. 
Models are saved whenever a new best validation score is reached, in `batch_no.ckpt`, where `batch_no` is the number of batches the model has been trained on so far.
`best.ckpt` links to the checkpoint that has so far achieved the best validation score.


#### Visualization
JoeyNMT uses Tensorboard to visualize training and validation curves and attention matrices during training.
Launch [Tensorboard](https://github.com/tensorflow/tensorboard) with `tensorboard --logdir model_dir/tensorboard` (or `python -m tensorboard.main ...`) and then open the url (default: `localhost:6006`) with a browser. 

For a stand-alone plot, run `python3 scripts/plot_validation.py model_dir --plot_values bleu PPL --output_path my_plot.pdf` to plot curves of validation BLEU and PPL.

#### CPU vs. GPU
For training on a GPU, set `use_cuda` in the config file to `True`. This requires the installation of required CUDA libraries.


### Translating

There are three options for testing what the model has learned.

Whatever data you feed the model for translating, make sure it is properly pre-processed, just as you pre-processed the training data, e.g. tokenized and split into subwords (if working with BPEs).

#### 1. Test Set Evaluation 
For testing and evaluating on your parallel test/dev set, run 

`python3 -m joeynmt test configs/small.yaml --output_path out`.

This will generate translations for validation and test set (as specified in the configuration) in `out.[dev|test]`
with the latest/best model in the `model_dir` (or a specific checkpoint set with `load_model`).
It will also evaluate the outputs with `eval_metric`.
If `--output_path` is not specified, it will not store the translation, and only do the evaluation and print the results.

#### 2. File Translation
In order to translate the contents of a file not contained in the configuration (here `my_input.txt`), simply run

`python3 -m joeynmt translate configs/small.yaml < my_input.txt > out`.

The translations will be written to stdout or alternatively`--output_path` if specified.

#### 3. Interactive
If you just want try a few examples, run

`python3 -m joeynmt translate configs/small.yaml`

and you'll be prompted to type input sentences that JoeyNMT will then translate with the model specified in the configuration.


## Documentation and Tutorial
[The docs](https://joeynmt.readthedocs.io) include an overview of the NMT implementation, a walk-through tutorial for building, training, tuning, testing and inspecting an NMT system, the [API documentation]() and [FAQs]().

A screencast of the tutorial is available on [YouTube](https://www.youtube.com/watch?v=PzWRWSIwSYc).

## Benchmarks
Benchmark results on WMT and IWSLT datasets are reported [here](benchmarks.md).

## Pre-trained Models
Pre-trained models from reported benchmarks for download (contains config, vocabularies, best checkpoint and dev/test hypotheses):
- [WMT17 en-de "best" RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_ende_best.tar.gz) (2G)
- [WMT17 lv-en "best" RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_lven_best.tar.gz) (1.9G)
- [WMT17 en-de Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_ende_transformer.tar.gz) (664M)
- [WMT17 lv-en Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_lven_transformer.tar.gz) (650M)

## Contributing
Since this codebase is supposed to stay clean and minimalistic, contributions addressing the following are welcome:
- Code correctness
- Code cleanliness
- Documentation quality
- Speed or memory improvements
- resolving issues

Code extending the functionalities beyond the basics will most likely not end up in the master branch, but we're curions to learn what you used Joey for.

## Projects and Extensions
Here we'll collect projects and repositories that are based on Joey, so you can find inspiration and examples on how to modify and extend the code.

- **User Study**. We evaluated the code quality of this repository by testing the understanding of novices through quiz questions. Find the details in Section 3 of the [Joey NMT paper](https://arxiv.org/abs/1907.12484).
- **Self-Regulated Interactive Seq2Seq Learning**. Julia Kreutzer and Stefan Riezler. Published at ACL 2019. [Paper](https://arxiv.org/abs/1907.05190) and [Code](https://github.com/juliakreutzer/joeynmt/tree/acl19). This project augments the standard fully-supervised learning regime by weak and self-supervision for a better trade-off of quality and supervision costs in interactive NMT.
- **Speech Joey**. [@Sariyusha](https://github.com/Sariyusha) is giving Joey ears for speech translation. [Code](https://github.com/Sariyusha/speech_joey). 

If you used Joey NMT for a project, publication or built some code on top of it, let us know and we'll link it here.


## Reference
If you use Joey NMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

```
@ARTICLE{JoeyNMT,
author = {{Kreutzer}, Julia and {Bastings}, Jasmijn and {Riezler}, Stefan},
title = {Joey {NMT}: A Minimalist {NMT} Toolkit for Novices},
journal = {To Appear in EMNLP-IJCNLP 2019: System Demonstrations},
year = {2019},
month = {Nov},
address = {Hong Kong}
url = {https://arxiv.org/abs/1907.12484}
}
```

## Naming
Joeys are [infant marsupials](https://en.wikipedia.org/wiki/Marsupial#Early_development). 

