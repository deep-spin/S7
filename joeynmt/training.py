# coding: utf-8

"""
Training module
"""

from itertools import count
import argparse
import time
import shutil
from typing import List, Dict
import os
from os.path import join
import queue
from functools import partial
import random

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset

from entmax import Entmax15Loss, SparsemaxLoss, EntmaxBisectLoss

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, postprocess
from joeynmt.model import Model
from joeynmt.prediction import validate_on_data
from joeynmt.loss import LabelSmoothingLoss, FYLabelSmoothingLoss
from joeynmt.data import load_data, make_data_iter
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.prediction import test


def _parse_hidden_size(model_config):
    if "encoder" in model_config:
        return model_config["encoder"]["hidden_size"]

    encs_config = model_config["encoders"]
    if "encoder" in encs_config:
        return encs_config["encoder"]["hidden_size"]
    return next(cf["hidden_size"] for cf in encs_config.values()
                if "hidden_size" in cf)


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: Model, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = train_config["model_dir"]
        make_model_dir(
            self.model_dir, overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = join(self.model_dir, "validations.txt")
        self.tb_writer = SummaryWriter(
            log_dir=join(self.model_dir, "tensorboard/")
        )

        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self._log_parameters_list()

        # objective
        objective = train_config.get("loss", "cross_entropy")
        loss_alpha = train_config.get("loss_alpha", 1.5)

        assert loss_alpha >= 1
        # maybe don't do the label smoothing thing here, instead have
        # nn.CrossEntropyLoss
        # then you look up the loss func, and you either use it directly or
        # wrap it in FYLabelSmoothingLoss
        if objective == "softmax":
            objective = "cross_entropy"
        loss_funcs = {
            "cross_entropy": nn.CrossEntropyLoss,
            "entmax15": partial(Entmax15Loss, k=512),
            "sparsemax": partial(SparsemaxLoss, k=512),
            "entmax": partial(EntmaxBisectLoss, alpha=loss_alpha, n_iter=30)
        }
        if objective not in loss_funcs:
            raise ConfigurationError("Unknown loss function")

        loss_module = loss_funcs[objective]
        loss_func = loss_module(ignore_index=self.pad_index, reduction='sum')

        label_smoothing = train_config.get("label_smoothing", 0.0)
        label_smoothing_type = train_config.get("label_smoothing_type", "fy")
        assert label_smoothing_type in ["fy", "szegedy"]
        smooth_dist = train_config.get("smoothing_distribution", "uniform")
        assert smooth_dist in ["uniform", "unigram"]
        if label_smoothing > 0:
            if label_smoothing_type == "fy":
                # label smoothing entmax loss
                if smooth_dist is not None:
                    smooth_p = torch.FloatTensor(model.trg_vocab.frequencies)
                    smooth_p /= smooth_p.sum()
                else:
                    smooth_p = None
                loss_func = FYLabelSmoothingLoss(
                    loss_func, smoothing=label_smoothing, smooth_p=smooth_p
                )
            else:
                assert objective == "cross_entropy"
                loss_func = LabelSmoothingLoss(
                    ignore_index=self.pad_index,
                    reduction="sum",
                    smoothing=label_smoothing
                )
        self.loss = loss_func

        self.norm_type = train_config.get("normalization", "batch")
        if self.norm_type not in ["batch", "tokens"]:
            raise ConfigurationError("Invalid normalization. "
                                     "Valid options: 'batch', 'tokens'.")

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters())

        # validation & early stopping
        self.validate_by_label = train_config.get("validate_by_label", False)
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.plot_attention = train_config.get("plot_attention", False)
        self.ckpt_queue = queue.Queue(
            maxsize=train_config.get("keep_last_ckpts", 5))

        allowed = {'bleu', 'chrf', 'token_accuracy',
                   'sequence_accuracy', 'cer', "wer", "levenshtein_distance"}
        eval_metrics = train_config.get("eval_metric", "bleu")
        if isinstance(eval_metrics, str):
            eval_metrics = [eval_metrics]
        if any(metric not in allowed for metric in eval_metrics):
            ok_metrics = " ".join(allowed)
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: {}".format(ok_metrics))
        self.eval_metrics = eval_metrics
        self.forced_sparsity = train_config.get("forced_sparsity", False)

        early_stop_metric = train_config.get("early_stopping_metric", "loss")
        allowed_early_stop = {"ppl", "loss"} | set(self.eval_metrics)
        if early_stop_metric not in allowed_early_stop:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', and eval_metrics.")
        self.early_stopping_metric = early_stop_metric
        min_metrics = {"ppl", "loss", "cer", "wer", "levenshtein_distance"}
        self.minimize_metric = early_stop_metric in min_metrics

        # learning rate scheduling
        hidden_size = _parse_hidden_size(config["model"])
        self.scheduler, self.sched_incr = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=hidden_size)

        # data & batch handling
        # src/trg magic
        if "level" in config["data"]:
            self.src_level = self.trg_level = config["data"]["level"]
        else:
            assert "src_level" in config["data"]
            assert "trg_level" in config["data"]
            self.src_level = config["data"]["src_level"]
            self.trg_level = config["data"]["trg_level"]

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size",
                                                self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",
                                                self.batch_type)

        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf

        mrt_schedule = train_config.get("mrt_schedule", None)
        assert mrt_schedule is None or mrt_schedule in ["warmup", "mix", "mtl"]
        self.mrt_schedule = mrt_schedule
        self.mrt_p = train_config.get("mrt_p", 0.0)
        self.mrt_lambda = train_config.get("mrt_lambda", 1.0)
        assert 0 <= self.mrt_p <= 1
        assert 0 <= self.mrt_lambda <= 1
        self.mrt_start_steps = train_config.get("mrt_start_steps", 0)
        self.mrt_samples = train_config.get("mrt_samples", 1)
        self.mrt_alpha = train_config.get("mrt_alpha", 1.0)
        self.mrt_strategy = train_config.get("mrt_strategy", "sample")
        self.mrt_cost = train_config.get("mrt_cost", "levenshtein")
        self.mrt_max_len = train_config.get("mrt_max_len", 31)  # hmm
        self.step_counter = count()

        assert self.mrt_alpha > 0
        assert self.mrt_strategy in ["sample", "topk"]
        assert self.mrt_cost in ["levenshtein", "bleu"]

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            reset_training = train_config.get("reset_training", False)
            self.logger.info("Loading model from %s", model_load_path)
            self.init_from_checkpoint(model_load_path, reset=reset_training)

    def is_best(self, score):
        return self.minimize_metric == (score < self.best_ckpt_score)

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        ckpt_name = str(self.steps) + ".ckpt"
        model_path = join(self.model_dir, ckpt_name)
        if self.scheduler is not None:
            scheduler_state = self.scheduler.state_dict()
        else:
            scheduler_state = None
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": scheduler_state
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning("Wanted to delete old checkpoint %s but "
                                    "file does not exist.", to_delete)

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update(ckpt_name, join(self.model_dir, "best.ckpt"))

    def init_from_checkpoint(self, path: str, reset: bool = False):
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
            scheduler_state = model_checkpoint["scheduler_state"]
            if scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_state)

            # restore counts
            self.steps = model_checkpoint["steps"]
            self.total_tokens = model_checkpoint["total_tokens"]
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def log_tensorboard(self, split, **kwargs):
        """
        split: "train" or "valid"
        """
        assert split in ["train", "valid"]
        prefix = "{0}/{0}_".format(split)
        for metric, value in kwargs.items():
            name = prefix + metric
            self.tb_writer.add_scalar(name, value, self.steps)

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset):
        """
        Train the model and validate it on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=True,
            shuffle=self.shuffle,
            use_cuda=self.use_cuda
        )
        for epoch_no in range(1, self.epochs + 1):
            self.logger.info("EPOCH %d", epoch_no)

            if self.sched_incr == "epoch":
                self.scheduler.step(epoch=epoch_no - 1)  # 0-based indexing

            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens
            epoch_loss = 0

            for i, batch in enumerate(iter(train_iter), 1):
                # reactivate training
                self.model.train()
                # create a Batch object from torchtext batch
                batch = Batch(batch, self.pad_index)

                # only update every batch_multiplier batches
                update = i % self.batch_multiplier == 0
                batch_loss = self._train_batch(batch, update=update)

                self.log_tensorboard("train", batch_loss=batch_loss)

                epoch_loss += batch_loss.cpu().numpy()

                if not update:
                    continue

                if self.sched_incr == "step":
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    processed_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0:
                    valid_start_time = time.time()

                    # it would be nice to include loss and ppl in valid_scores
                    valid_scores, valid_references, valid_hypotheses, \
                        valid_hypotheses_raw, valid_attention_scores, \
                        scores_by_label = validate_on_data(
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metrics=self.eval_metrics,
                            trg_level=self.trg_level,
                            model=self.model,
                            use_cuda=self.use_cuda,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            beam_size=0,  # greedy validations
                            batch_type=self.eval_batch_type,
                            save_attention=self.plot_attention,
                            validate_by_label=self.validate_by_label,
                            forced_sparsity=self.forced_sparsity
                        )

                    ckpt_score = valid_scores[self.early_stopping_metric]
                    self.log_tensorboard("valid", **valid_scores)

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if self.sched_incr == "validation":
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(valid_scores, new_best=new_best)

                    valid_sources_raw = {f: list(getattr(valid_data, f))
                                         for f in valid_data.fields
                                         if f != "trg"}

                    self._log_examples(
                        sources_raw=valid_sources_raw,
                        hypotheses_raw=valid_hypotheses_raw,
                        hypotheses=valid_hypotheses,
                        references=valid_references
                    )

                    labeled_scores = sorted(valid_scores.items())
                    eval_report = ", ".join("{}: {:.5f}".format(n, v)
                                            for n, v in labeled_scores)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration

                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: %s, '
                        'duration: %.4fs',
                        epoch_no, self.steps, eval_report, valid_duration)

                    if scores_by_label is not None:
                        for metric, scores in scores_by_label.items():
                            # make a report
                            label_report = [metric]
                            numbers = sorted(scores.items())
                            label_report.extend(
                                ["{}={}: {:.5f}".format(l, n, v)
                                 for (l, n), v in numbers]
                            )
                            self.logger.info("\n\t".join(label_report))

                    # store validation set outputs
                    self._store_outputs(valid_hypotheses)

                    # store attention plots for selected valid sentences
                    if valid_attention_scores and self.plot_attention:
                        store_attention_plots(
                                attentions=valid_attention_scores,
                                sources=list(valid_data.src),
                                targets=valid_hypotheses_raw,
                                indices=self.log_valid_sents,
                                model_dir=self.model_dir,
                                tb_writer=self.tb_writer,
                                steps=self.steps)

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended because minimum lr %f was reached.',
                    self.learning_rate_min)
                break

            self.logger.info(
                'Epoch %3d: total training loss %.2f', epoch_no, epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return: loss for batch (sum)
        """
        times_called = next(self.step_counter)
        # when do you call get_risk_for batch?
        mrt_schedule = self.mrt_schedule
        mrt_steps = self.mrt_start_steps
        warmed_up = mrt_schedule == "warmup" and times_called >= mrt_steps
        mrt_drawn = mrt_schedule == "mix" and random.random() < self.mrt_p

        if mrt_schedule == "mtl":
            batch_loss = self.model.get_loss_and_risk_for_batch(
                batch,
                self.loss,
                n_samples=self.mrt_samples,
                alpha=self.mrt_alpha,
                strategy=self.mrt_strategy,
                max_len=self.mrt_max_len,
                cost=self.mrt_cost,
                level=self.trg_level,
                mrt_lambda=self.mrt_lambda
            )
        if warmed_up or mrt_drawn:
            batch_loss = self.mrt_lambda * self.model.get_risk_for_batch(
                batch,
                n_samples=self.mrt_samples,
                alpha=self.mrt_alpha,
                strategy=self.mrt_strategy,
                max_len=self.mrt_max_len,
                cost=self.mrt_cost,
                level=self.trg_level
            )
        else:
            batch_loss = self.model.get_loss_for_batch(batch, self.loss)

        norm = batch.nseqs if self.norm_type == "batch" else batch.ntokens

        norm_batch_loss = batch_loss / norm
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            self.clip_grad_fun(self.model.parameters())  # works in-place

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1

        # increment token counter
        self.total_tokens += batch.ntokens

        return norm_batch_loss.detach()

    def _add_report(self, valid_scores: dict, new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True  # why does this happen inside _add_report?

        with open(self.valid_report_file, 'a') as opened_file:
            labeled_scores = sorted(valid_scores.items())
            eval_report = "\t".join("{}: {:.5f}".format(n, v)
                                    for n, v in labeled_scores)
            opened_file.write(
                "Steps: {}\t{}\tLR: {:.8f}\t{}\n".format(
                    self.steps, eval_report,
                    current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum(np.prod(p.size()) for p in model_parameters)
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
            self,
            sources_raw: Dict[str, List[str]],
            hypotheses: List[str],
            references: List[str],
            hypotheses_raw: List[List[str]] = None,
            references_raw: List[List[str]] = None) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (dict of list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        """
        ix = self.log_valid_sents
        assert all(i < len(hypotheses) for i in ix)

        sources = {k: postprocess(v, self.src_level)
                   for k, v in sources_raw.items()}
        for i in ix:
            self.logger.info("Example #{}".format(i))

            for f, rs in sources_raw.items():
                self.logger.debug("\t{}:     {}".format(f, rs[i]))
            if references_raw is not None:
                self.logger.debug("\tRaw reference:  %s", references_raw[i])
            if hypotheses_raw is not None:
                self.logger.debug("\tRaw hypothesis: %s", hypotheses_raw[i])

            for f, srcs in sources.items():
                self.logger.info("\t{}:     {}".format(f, srcs[i]))
            self.logger.info("\tReference:  %s", references[i])
            self.logger.info("\tHypothesis: %s", hypotheses[i])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        valid_output_file = join(self.model_dir, "{}.hyps".format(self.steps))
        with open(valid_output_file, 'w') as f:
            for hyp in hypotheses:
                f.write("{}\n".format(hyp))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # set the random seed
    set_seed(seed=train_cfg.get("random_seed", 42))

    # load the data
    data = load_data(data_cfg)
    train_data = data["train_data"]
    dev_data = data["dev_data"]
    test_data = data["test_data"]
    vocabs = data["vocabs"]

    # build an encoder-decoder model
    model = build_model(cfg["model"], vocabs=vocabs)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, join(trainer.model_dir, "config.yaml"))

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        test_data=test_data,
        vocabs=vocabs,
        logging_function=trainer.logger.info)

    trainer.logger.info(str(model))

    # store the vocabs
    model_dir = train_cfg["model_dir"]
    for vocab_name, vocab in vocabs.items():
        vocab.to_file(join(model_dir, vocab_name + "_vocab.txt"))

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = join(trainer.model_dir, str(trainer.best_ckpt_iteration) + ".ckpt")
    output_name = "{:08d}.hyps".format(trainer.best_ckpt_iteration)
    output_path = join(trainer.model_dir, output_name)
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=trainer.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
