# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
import logging
from typing import Optional, Sequence
from functools import partial
from collections import defaultdict
import math

import torch
from torchtext.data import Dataset, Field

from joeynmt.helpers import postprocess, load_config, ConfigurationError, \
    get_latest_checkpoint, load_checkpoint, store_attention_plots
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy, \
    character_error_rate, word_error_rate, levenshtein_distance
from joeynmt.model import build_model, Model, EnsembleModel
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter, MonoDataset
from joeynmt.datasets import TSVDataset
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary


def len_penalty(score, step, alpha):
    length_penalty = ((5.0 + step) / 6.0) ** alpha
    return score / length_penalty


def expected_calibration_error(corrects, confidences, n_bins=10):
    corrects = torch.cat(corrects)
    confidences = torch.cat(confidences)
    n = corrects.size(0)
    bin_sizes = torch.histc(confidences, bins=n_bins, min=0, max=1).long()
    sorted_p, sorted_ix = torch.sort(confidences)
    sorted_corrects = corrects[sorted_ix]
    end_ix = bin_sizes.cumsum(dim=0)
    start_ix = end_ix - bin_sizes
    ece = 0.0
    for start_i, end_i in zip(start_ix, end_ix):
        if end_i > start_i:
            bin_size = (end_i - start_i).item()
            bin_p = sorted_p[start_i: end_i]
            bin_corrects = sorted_corrects[start_i: end_i]
            acc_bm = bin_corrects.sum().item() / bin_size
            conf_bm = bin_p.sum().item() / bin_size
            ece += bin_size / n * abs(acc_bm - conf_bm)
    return ece


def evaluate_decoding(
        data, valid_refs, valid_hyps, eval_metrics, validate_by_label):
    """
    Evaluate the model by computing metrics (given in eval_metrics) between
    reference sequences and hypotheses decoded from the model.
    Returns:
        - a dict whose keys are the names of metrics (one per element of
          eval_metrics)
        - another dict (or None)
    """
    valid_scores = dict()  # what we'll eventually return

    # are you macro-averaging over labels? If so, then split them here
    by_label = defaultdict(list)
    if validate_by_label:
        label_values = defaultdict(list)
        for label in data.label_columns:
            label_values[label].extend(lab for lab in getattr(data, label))
        for label, values in label_values.items():
            seqs = zip(valid_refs, valid_hyps) if valid_refs else valid_hyps
            examples = zip(values, seqs)
            for lang, seq in examples:
                by_label[label, lang].append(seq)
    else:
        seqs = zip(valid_refs, valid_hyps) if valid_refs else valid_hyps
        by_label[None].extend(seqs)

    # if references are given, evaluate against them
    scores_by_label = dict()
    if valid_refs and eval_metrics is not None:
        assert len(valid_hyps) == len(valid_refs)

        for eval_metric, eval_func in eval_metrics.items():
            score_by_label = dict()
            for label, pairs in by_label.items():
                label_hyps, label_refs = zip(*pairs)
                label_score = eval_func(label_hyps, label_refs)
                score_by_label[label] = label_score

            score = sum(score_by_label.values()) / len(score_by_label)
            valid_scores[eval_metric] = score
            scores_by_label[eval_metric] = score_by_label

    if not validate_by_label:
        scores_by_label = None
    return valid_scores, scores_by_label


def validate_on_data(model: Model, data: Dataset,
                     batch_size: int,
                     use_cuda: bool,
                     max_output_length: int,
                     trg_level: str,
                     eval_metrics: Optional[Sequence[str]],
                     loss_function: torch.nn.Module = None,
                     beam_size: int = 0,
                     force_prune_size: int = 5,
                     beam_alpha: int = 0,
                     batch_type: str = "sentence",
                     save_attention: bool = False,
                     validate_by_label: bool = False,
                     forced_sparsity: bool = False,
                     method=None,
                     max_hyps=1,
                     break_at_p: float = 1.0,
                     break_at_argmax: bool = False,
                     short_depth: int = 0):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model:
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda:
    :param max_output_length: maximum length for generated hypotheses
    :param trg_level: target segmentation level
    :param eval_metrics:
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation (default 0 is greedy)
    :param beam_alpha: beam search alpha for length penalty (default 0)
    :param batch_type: validation batch type (sentence or token)

    :return:
        - current_valid_scores: current validation score [eval_metric],
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    if beam_size > 0:
        force_prune_size = beam_size

    if validate_by_label:
        assert isinstance(data, TSVDataset) and data.label_columns

    valid_scores = defaultdict(float)  # container for scores
    stats = defaultdict(float)

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False, use_cuda=use_cuda)

    pad_index = model.trg_vocab.stoi[PAD_TOKEN]

    model.eval()  # disable dropout

    force_objectives = loss_function is not None or forced_sparsity

    # possible tasks are: force w/ gold, force w/ empty, search
    scorer = partial(len_penalty, alpha=beam_alpha) if beam_alpha > 0 else None
    confidences = []
    corrects = []
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = defaultdict(list)
        for valid_batch in iter(valid_iter):
            batch = Batch(valid_batch, pad_index)
            rev_index = batch.sort_by_src_lengths()

            encoder_output, _ = model.encode(batch)

            empty_probs = None
            if force_objectives and not isinstance(model, EnsembleModel):
                # compute all the logits.
                logits = model.force_decode(batch, encoder_output)[0]
                bsz, gold_len, vocab_size = logits.size()
                gold, gold_lengths, _ = batch["trg"]
                prediction_steps = gold_lengths.sum().item() - bsz
                assert gold.size(0) == bsz

                if loss_function is not None:
                    gold_pred = gold[:, 1:].contiguous().view(-1)
                    batch_loss = loss_function(
                        logits.view(-1, logits.size(-1)), gold_pred
                    )
                    valid_scores["loss"] += batch_loss

                if forced_sparsity:
                    # compute probabilities
                    out = logits.view(-1, vocab_size)
                    if isinstance(model, EnsembleModel):
                        probs = out
                    else:
                        probs = model.decoder.gen_func(out, dim=-1)

                    # Compute numbers derived from the distributions.
                    # This includes support size, entropy, and calibration
                    non_pad = (gold[:, 1:] != pad_index).view(-1)
                    real_probs = probs[non_pad]
                    n_supported = real_probs.gt(0).sum().item()
                    pred_ps, pred_ix = real_probs.max(dim=-1)
                    real_gold = gold[:, 1:].contiguous().view(-1)[non_pad]
                    real_correct = pred_ix.eq(real_gold)
                    corrects.append(real_correct)
                    confidences.append(pred_ps)

                    beam_probs, _ = real_probs.topk(force_prune_size, dim=-1)
                    pruned_mass = 1 - beam_probs.sum(dim=-1)
                    stats["force_pruned_mass"] += pruned_mass.sum().item()

                    # compute stuff with the empty sequence
                    empty_probs = probs.view(
                        bsz, gold_len, vocab_size
                    )[:, 0, model.eos_index]
                    assert empty_probs.size() == gold_lengths.size()
                    empty_possible = empty_probs.gt(0).sum().item()
                    empty_mass = empty_probs.sum().item()

                    stats["eos_supported"] += empty_possible
                    stats["eos_mass"] += empty_mass
                    stats["n_supp"] += n_supported
                    stats["n_pred"] += prediction_steps

                short_scores = None
                if short_depth > 0:
                    # we call run_batch again with the short depth. We don't
                    # really care what the hypotheses are, we only want the
                    # scores
                    _, _, short_scores = model.run_batch(
                        batch=batch,
                        beam_size=beam_size,  # can this be removed?
                        scorer=scorer,  # should be none
                        max_output_length=short_depth,
                        method="dfs",
                        max_hyps=max_hyps,
                        encoder_output=encoder_output,
                        return_scores=True)

            # run as during inference to produce translations
            # todo: return_scores for greedy
            output, attention_scores, beam_scores = model.run_batch(
                batch=batch, beam_size=beam_size, scorer=scorer,
                max_output_length=max_output_length, method=method,
                max_hyps=max_hyps, encoder_output=encoder_output,
                return_scores=True, break_at_argmax=break_at_argmax,
                break_at_p=break_at_p)
            stats["hyp_length"] += output.ne(model.pad_index).sum().item()
            if beam_scores is not None and empty_probs is not None:
                # I need to expand this to handle stuff up to length m.
                # note that although you can compute the probability of the
                # empty sequence without any extra computation, you *do* need
                # to do extra decoding if you want to get the most likely
                # sequence with length <= m.
                empty_better = empty_probs.log().gt(beam_scores).sum().item()
                stats["empty_better"] += empty_better

                if short_scores is not None:
                    short_better = short_scores.gt(beam_scores).sum().item()
                    stats["short_better"] += short_better

            # sort outputs back to original order
            all_outputs.extend(output[rev_index])

            if save_attention and attention_scores is not None:
                # beam search currently does not support attention logging
                for k, v in attention_scores.items():
                    valid_attention_scores[k].extend(v[rev_index])

        assert len(all_outputs) == len(data)

    ref_length = sum(len(d.trg) for d in data)
    valid_scores["length_ratio"] = stats["hyp_length"] / ref_length

    assert len(corrects) == len(confidences)
    if corrects:
        valid_scores["ece"] = expected_calibration_error(corrects, confidences)

    if stats["n_pred"] > 0:
        valid_scores["ppl"] = math.exp(valid_scores["loss"] / stats["n_pred"])

    if forced_sparsity and stats["n_pred"] > 0:
        valid_scores["support"] = stats["n_supp"] / stats["n_pred"]
        valid_scores["empty_possible"] = stats["eos_supported"] / len(all_outputs)
        valid_scores["empty_prob"] = stats["eos_mass"] / len(all_outputs)
        valid_scores["force_pruned_mass"] = stats["force_pruned_mass"] / stats["n_pred"]
        if beam_size > 0:
            valid_scores["empty_better"] = stats["empty_better"] / len(all_outputs)
            if short_depth > 0:
                score_name = "depth_{}_better".format(short_depth)
                valid_scores[score_name] = stats["short_better"] / len(all_outputs)

    # postprocess
    raw_hyps = model.trg_vocab.arrays_to_sentences(all_outputs)
    valid_hyps = postprocess(raw_hyps, trg_level)
    valid_refs = postprocess(data.trg, trg_level)

    # evaluate
    eval_funcs = {
        "bleu": bleu,
        "chrf": chrf,
        "token_accuracy": partial(token_accuracy, level=trg_level),
        "sequence_accuracy": sequence_accuracy,
        "wer": word_error_rate,
        "cer": partial(character_error_rate, level=trg_level),
        "levenshtein_distance": partial(levenshtein_distance, level=trg_level)
    }
    selected_eval_metrics = {name: eval_funcs[name] for name in eval_metrics}
    decoding_scores, scores_by_label = evaluate_decoding(
        data, valid_refs, valid_hyps, selected_eval_metrics, validate_by_label
    )
    valid_scores.update(decoding_scores)

    return valid_scores, valid_refs, valid_hyps, \
        raw_hyps, valid_attention_scores, scores_by_label


def log_scores(logger, name, scores, scores_by_label, beam_size, beam_alpha):
    if beam_size == 0:
        dec_log = "Greedy decoding"
    else:
        dec_log = "Beam search decoding with size = {} and alpha = {}".format(
            beam_size, beam_alpha
        )
    labeled_scores = sorted(scores.items())
    report = ", ".join("{}: {:.5f}".format(n, v) for n, v in labeled_scores)
    logger.info("%4s %s: [%s]", name, report, dec_log)
    if scores_by_label is not None:
        for metric, sc in scores_by_label.items():
            label_report = [metric]
            label_report.extend(
                ["{}={}: {:.5f}".format(l, n, v)
                 for (l, n), v in sorted(sc.items())]
            )
            logger.info("\n\t".join(label_report))


def test(cfg_file,
         ckpt,
         output_path: str = None,
         save_attention: bool = False,
         logger: logging.Logger = None, data_to_test: str = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        FORMAT = '%(asctime)-15s - %(message)s'
        logging.basicConfig(format=FORMAT)
        logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    test_cfg = cfg["testing"]

    if "test" not in data_cfg.keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    model_dir = train_cfg["model_dir"]
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError("No checkpoint at {}.".format(model_dir))
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    batch_size = train_cfg.get("eval_batch_size", train_cfg["batch_size"])
    batch_type = train_cfg.get("batch_type", "sentence")
    use_cuda = train_cfg.get("use_cuda", False)
    assert "level" in data_cfg or "trg_level" in data_cfg
    trg_level = data_cfg.get("level", data_cfg["trg_level"])

    eval_metric = train_cfg["eval_metric"]
    if isinstance(eval_metric, str):
        eval_metric = [eval_metric]
    max_output_length = test_cfg.get(
        "max_output_length", train_cfg.get("max_output_length", None)
    )

    # load the data
    data = load_data(data_cfg)
    dev_data = data["dev_data"]
    test_data = data["test_data"]
    vocabs = data["vocabs"]

    data_to_predict = {"dev": dev_data, "test": test_data}
    if data_to_test is not None:
        assert data_to_test in data_to_predict
        data_to_predict = {data_to_test: data_to_predict[data_to_test]}

    # load model state from disk
    if isinstance(ckpt, str):
        ckpt = [ckpt]
    models = []
    for c in ckpt:
        model_checkpoint = load_checkpoint(c, use_cuda=use_cuda)

        # build model and load parameters into it
        m = build_model(cfg["model"], vocabs=vocabs)
        m.load_state_dict(model_checkpoint["model_state"])
        models.append(m)
    model = models[0] if len(models) == 1 else EnsembleModel(*models)

    if use_cuda:
        model.cuda()  # should this exist?

    # whether to use beam search for decoding, 0: greedy decoding
    beam_sizes = beam_alpha = 0
    if "testing" in cfg.keys():
        beam_sizes = test_cfg.get("beam_size", 0)
        beam_alpha = test_cfg.get("alpha", 0)
    beam_sizes = [beam_sizes] if isinstance(beam_sizes, int) else beam_sizes
    assert beam_alpha >= 0, "Use alpha >= 0"

    method = test_cfg.get("method", None)
    max_hyps = test_cfg.get("max_hyps", 1)  # only for the enumerate thing

    validate_by_label = test_cfg.get(
        "validate_by_label", train_cfg.get("validate_by_label", False)
    )
    forced_sparsity = test_cfg.get(
        "forced_sparsity", train_cfg.get("forced_sparsity", False)
    )

    for beam_size in beam_sizes:
        for data_set_name, data_set in data_to_predict.items():
            valid_results = validate_on_data(
                model,
                data=data_set,
                batch_size=batch_size,
                batch_type=batch_type,
                trg_level=trg_level,
                max_output_length=max_output_length,
                eval_metrics=eval_metric,
                use_cuda=use_cuda,
                loss_function=None,
                beam_size=beam_size,
                beam_alpha=beam_alpha,
                save_attention=save_attention,
                validate_by_label=validate_by_label,
                forced_sparsity=forced_sparsity,
                method=method,
                max_hyps=max_hyps,
                break_at_p=test_cfg.get("break_at_p", 1.0),
                break_at_argmax=test_cfg.get("break_at_argmax", False),
                short_depth=test_cfg.get("short_depth", 0)
            )
            scores = valid_results[0]
            hypotheses, hypotheses_raw = valid_results[2:4]
            scores_by_label = valid_results[5]

            if "trg" in data_set.fields:
                log_scores(
                    logger, data_set_name,
                    scores, scores_by_label,
                    beam_size, beam_alpha
                )
            else:
                logger.info("No references given for %s -> no evaluation.",
                            data_set_name)

            attention_scores = valid_results[4]
            if save_attention and not attention_scores:
                logger.warning("Attention scores could not be saved. "
                               "Note that attention scores are not "
                               "available when using beam search. "
                               "Set beam_size to 0 for greedy decoding.")
            if save_attention and attention_scores:
                # currently this will break for transformers
                logger.info("Saving attention plots. This might be slow.")
                store_attention_plots(attentions=attention_scores,
                                      targets=hypotheses_raw,
                                      sources=[s for s in data_set.src],
                                      indices=range(len(hypotheses)),
                                      model_dir=model_dir,
                                      steps=step,
                                      data_set_name=data_set_name)
                logger.info("Attention plots saved to: %s", model_dir)

            if output_path is not None:
                output_path_set = "{}.{}".format(output_path, data_set_name)
                with open(output_path_set, mode="w", encoding="utf-8") as outf:
                    for hyp in hypotheses:
                        outf.write(hyp + "\n")
                logger.info("Translations saved to: %s", output_path_set)


def translate(cfg_file, ckpt: str, output_path: str = None) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    """

    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name+tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix, field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        hypotheses = validate_on_data(
            model, data=test_data, batch_size=batch_size, trg_level=level,
            max_output_length=max_output_length, eval_metrics=[],
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha)[2]
        return hypotheses

    cfg = load_config(cfg_file)

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)

    batch_size = cfg["training"].get("batch_size", 1)
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # read vocabs
    src_vocab_file = cfg["data"].get(
        "src_vocab", cfg["training"]["model_dir"] + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get(
        "trg_vocab", cfg["training"]["model_dir"] + "/trg_vocab.txt")
    src_vocab = Vocabulary.from_file(src_vocab_file)
    trg_vocab = Vocabulary.from_file(trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = list if level == "char" else str.split

    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(
        cfg["model"], vocabs={"src": src_vocab, "trg": trg_vocab}
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", 0)
    else:
        beam_size = 0
        beam_alpha = 0
    if beam_alpha < 0:
        raise ConfigurationError("alpha for length penalty should be >= 0")

    if not sys.stdin.isatty():
        # file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            print("Translations saved to: {}".format(output_path_set))
        else:
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
