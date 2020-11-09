import copy
import logging
import math
import os
import re
import shutil
import time
from itertools import cycle
from typing import Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, KLDivLoss
from torch.nn.functional import log_softmax
from torch.nn.parallel import DataParallel
from torch.nn.utils import clip_grad_norm

from openkge.dataset import OneToNMentionRelationDataset, EntityRelationDatasetBase
from openkge.model import RelationScorer, RelationEmbedder
from utils.log import ResultsLog
from utils.metrics import AccumulateMeter, MetricResult
from utils.optim import OptimRegime


def running_mean(new, old=None, momentum=0.9):
    if old is None:
        return new
    else:
        return momentum * old + (1 - momentum) * new


class AddLossModule(nn.Module):
    """
    Wraps around a module to add a loss for parallelization.
    """

    def __init__(
            self,
            model: Union[RelationScorer, RelationEmbedder],
            loss,
            bce_label_smoothing=0.0,
    ):
        super(AddLossModule, self).__init__()
        self.model : Union[RelationScorer, RelationEmbedder] = model
        self.loss = loss
        self.bce_label_smoothing = bce_label_smoothing

    def forward(
        self,
        inputs,
        labels,
        use_batch_shared_entities,
        batch_shared_entities,
        epoch=-1,
        input_style_triple_or_prefix="triple",
    ):

        allowed_input_style_triple_or_prefixes = ["triple", "right_and_left_prefix"]
        if input_style_triple_or_prefix not in allowed_input_style_triple_or_prefixes:
            raise Exception("input_style_triple_or_prefix not in {}".format(allowed_input_style_triple_or_prefixes))

        hook_loss = None

        if input_style_triple_or_prefix == "right_and_left_prefix":

            all_outputs = list()
            precomputed_batch_shared_entities = None

            for prefix_score_func, prefix_input in zip(
                [self.model.po_prefix_score, self.model.sp_prefix_score], inputs
            ):

                if prefix_input is not None:

                    if batch_shared_entities is not None \
                            and precomputed_batch_shared_entities is None:
                        if not use_batch_shared_entities and not self.model.training:
                            precomputed_batch_shared_entities = self.model.get_all_obj()
                        else:
                            precomputed_batch_shared_entities = self.model.precompute_batch_shared_inputs(
                                batch_shared_entities.view(-1),
                            )

                    if precomputed_batch_shared_entities is not None:
                        output = prefix_score_func(*prefix_input, precomputed_batch_shared_entities)
                    else:
                        output = prefix_score_func(*prefix_input)

                    all_outputs.append(output)

            all_outputs = torch.cat(all_outputs)

            if isinstance(self.loss, BCEWithLogitsLoss) or isinstance(self.loss, KLDivLoss):
                all_outputs_predictions = all_outputs

                if hasattr(self.model, "after_batch_loss_hook"):
                    hook_loss = self.model.after_batch_loss_hook(epoch)

                if isinstance(self.loss, nn.KLDivLoss):
                    all_outputs_predictions = log_softmax(all_outputs, dim=1)
                    # labels.data = labels.data/labels.data.sum(dim=-1)
                else:
                    if self.bce_label_smoothing > 0:
                        labels.data = labels.data + (1 / labels.data.size(-1))
                        labels.data = labels.data * (1 - self.bce_label_smoothing)
                result = self.loss(all_outputs_predictions.view(-1), labels.view(-1))
                return (
                    result,
                    hook_loss,
                    all_outputs
                )
            else:
                raise NotImplementedError(f"{self.loss} not supported. Please choose either BCEWithLogitsLoss or KLDivLoss")

class Trainer(object):

    def __init__(
        self,
        args,
        model: Union[RelationScorer, RelationEmbedder],
        loss,
        train_dataset: EntityRelationDatasetBase,
        validation_dataset: EntityRelationDatasetBase,
        train_loader,
        save_path=".",
        checkpoint_filename="checkpoint%s.pth.tar",
        keep_checkpoints=5,
    ):
        super(Trainer, self).__init__()

        self.args = args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.use_gpu = not args["no_cuda"]
        torch.set_num_threads(os.cpu_count())
        logging.info("Using GPU: {} || Number of CPU cores for torch {}".format(self.use_gpu, torch.get_num_threads()))

        self.optimizers = OptimRegime.setup_optimizer_regime(args=args, model=model)
        self.model = model
        self.loss = loss
        self.model_with_loss = AddLossModule(self.model, self.loss, args["bce_label_smoothing"])
        if (isinstance(args["devices"], list) or isinstance(args["devices"], tuple)) and len(args["devices"]) > 1:
            logging.info("Using DataParallel")
            self.model_with_loss = DataParallel(self.model_with_loss, args["devices"])

        self.checkpoint_filename = checkpoint_filename
        self.save_path = save_path

        self.save_info = {"config": args}
        self.results = ResultsLog(os.path.join(self.save_path, "results.{}".format("csv")))

        self.training_steps = 0
        self.len_train_batches = 1 if train_loader is None else len(train_loader)
        self.keep_checkpoints = keep_checkpoints
        self.counter = cycle(range(self.keep_checkpoints))
        self.save_epoch_freq_cycle = cycle(range(args["save_epoch_freq"]))
        self.save_epoch_freq = 0

        self.terminate = False
        self.terminate_epochs = 0 + args["patience_epochs"]
        self.best_validation_results = MetricResult()
        self.last_validation_metric = None
        self.moving_average_metric_change = None

        self.batch_size_for_backward = (
            self.train_dataset.batch_size_for_backward
            if self.train_dataset.batch_size_for_backward is not None
            else self.train_dataset.batch_size
        )
        self.batch_size_for_backward_accumulated = 0

    @property
    def epoch(self):
        return math.floor(self.training_steps / (self.len_train_batches + 1)) + 1

    @property
    def batch_first(self):
        return getattr(self.model, "batch_first", False)

    def compute_one_batch(self, data, training=True):

        # compute output

        backward_loss = None

        if training:
            data_set = self.train_dataset
        else:
            data_set = self.validation_dataset

        self.model_with_loss.to(data_set.device)

        inputs, \
        normalizer_loss, \
        normalizer_metric, \
        labels, \
        label_ids, \
        filter_mask, \
        batch_shared_entities = data_set.input_and_labels_to_device(
            data,
            training=training,
            device=data_set.device if not isinstance(self.model_with_loss, DataParallel) else "cpu",
        )

        loss, hook_loss, predictions = self.model_with_loss(
            inputs=inputs,
            labels=labels,
            batch_shared_entities=batch_shared_entities,
            use_batch_shared_entities=data_set.use_batch_shared_entities,
            epoch=self.epoch,
            input_style_triple_or_prefix=data_set.input_style,
        )

        batch_size = len(labels)

        if loss is not None:
            backward_loss = loss.sum()
            if hook_loss is not None:
                backward_loss = backward_loss + hook_loss
            backward_loss = backward_loss / normalizer_loss

        if training:

            # do the backward

            if backward_loss is not None:

                if self.batch_size_for_backward_accumulated == 0:
                    for optimizer in self.optimizers:
                        # compute gradient and do SGD step
                        optimizer.zero_grad()

                backward_loss.backward()

                self.batch_size_for_backward_accumulated += batch_size

                if self.batch_size_for_backward_accumulated == self.batch_size_for_backward:

                    for optimizer in self.optimizers:
                        if self.args["grad_clip"] is not None and self.args["grad_clip"] > 0:
                            clip_grad_norm(self.model.parameters(), self.args["grad_clip"])
                        optimizer.step()
                        optimizer.zero_grad()

                    self.batch_size_for_backward_accumulated = 0

                    metric_result = MetricResult()
                    metric_result["loss"].update(
                        loss.detach().item()/normalizer_loss if loss is not None else 0,
                        normalizer_loss
                    )
                    return metric_result, normalizer_metric,

                else:

                    return None, normalizer_metric,

        else:

            # compute the performance metrics over the pos + negative samples

            metric_result = OneToNMentionRelationDataset.compute_metrics(
                filter_mask,
                label_ids,
                predictions,
            )
            metric_result["loss"].update(
                loss.detach().item()/normalizer_loss if loss is not None else 0,
                normalizer_loss
            )
            return metric_result, normalizer_metric,

    def compute_one_epoch(
        self, data_loader, yield_result_steps=None, training=True, save=True,
    ):

        yield_result_steps = yield_result_steps or len(data_loader) - 1

        data_time_meter = AccumulateMeter("data_time_meter")
        items_per_time_meter = AccumulateMeter("items_per_time_meter")

        total_start = time.time()
        batch_start = time.time()

        metric_result = MetricResult()

        for step, batch in enumerate(data_loader):

            # print("for i, data in enumerate(data_loader) {}".format(time.time() - batch_start))
            # measure data loading time
            data_time_meter.update(time.time() - batch_start)

            if training:
                self.training_steps += 1
                self.len_train_batches = len(data_loader)
                # update optimizer according to epoch and steps
                for optimizer in self.optimizers:
                    optimizer.update(self.epoch, self.training_steps)

            # do an iteration
            new_metric_result, \
            normalizer_metric = self.compute_one_batch(
                batch, training=training
            )

            # measure elapsed time
            items_per_time_meter.update(normalizer_metric / (time.time() - batch_start), time.time() - batch_start)

            metric_result = metric_result + new_metric_result

            # end = time.time()
            last_step_of_epoch = step == len(data_loader) - 1

            if step > 0 or last_step_of_epoch:

                if step % self.args["print_freq"] == 0 or step % yield_result_steps == 0 or last_step_of_epoch:
                    # if training and step % self.print_freq == 0 or step % num_iterations == 0 or last_step_of_epoch:
                    logging.info(
                        "{phase} - EPOCH  [{epoch: >3}][{step: >6}/{total}]   "
                        "time: {batch_time: >7.3f}   "
                        "data: {data_time.avg:2.3f}   "
                        "items/sec: ({tok_time.avg:,.0f})   "
                        "METRICS  {metric_result.averages}".format(
                            epoch=self.epoch,
                            step=step,
                            total=len(data_loader),
                            str_len_total=len(str(len(data_loader))),
                            phase="TRAINING" if training else "EVALUATING",
                            batch_time=time.time() - total_start,
                            data_time=data_time_meter,
                            tok_time=items_per_time_meter,
                            metric_result=metric_result
                        )
                    )
                if save and (
                    training
                    and self.args["save_freq"] is not None
                    and self.args["save_freq"] > 0
                    and (step % self.args["save_freq"] == 0)
                    and step > 0
                ):
                    self.save(identifier=next(self.counter))

                if yield_result_steps > 0 and step % yield_result_steps == 0 or last_step_of_epoch:
                    yield copy.deepcopy(metric_result), \
                        {
                            "last_step": step,
                            "last_step_of_eval_freq": self.args["eval_freq"] > 0 and step % self.args["eval_freq"] == 0,
                            "last_step_of_epoch": last_step_of_epoch,
                        }
                    metric_result.reset()

            batch_start = time.time()

    def train(self, data_loader, yield_result_steps=None):
        # switch to train mode
        self.model_with_loss.train()
        for result in self.compute_one_epoch(data_loader, yield_result_steps=yield_result_steps, training=True):
            yield result
            self.model.train()

    def evaluate(self, data_loader, save=True):
        # switch to evaluate mode
        self.model_with_loss.eval()
        with torch.no_grad():
            for _result in self.compute_one_epoch(data_loader, training=False, save=save):
                result = _result
            return result

    def run(self, train_loader, val_loader=None):

        self.save_epoch_freq = next(self.save_epoch_freq_cycle)

        # loop over the training data

        for train_results, train_epoch_info in self.train(data_loader=train_loader, yield_result_steps=self.args["eval_freq"]):

            results = {
                "epoch": self.epoch,
                "training_steps": self.training_steps,
                "training_loss": train_results["loss"],
            }

            # after each epoch evaluate

            if val_loader is not None and (
                train_epoch_info["last_step_of_eval_freq"]
                or self.args["eval_epoch_freq"] is not None
                and self.args["eval_epoch_freq"] > 0
                and (self.epoch % self.args["eval_epoch_freq"]) == 0
            ):

                # run evaluations and collect results
                validation_results, _ = self.evaluate(val_loader)

                # check for early stopping
                one_metric_improved = False
                metric_improved = dict()
                best_select_metric = list()

                for metric_name, new_val_metric in validation_results.items():
                    metric_improved[metric_name] = False
                    if new_val_metric.avg_better_than(self.best_validation_results[metric_name]):
                        if metric_name in self.args["model_select_metric"]:
                            best_select_metric.append(metric_name)
                            one_metric_improved = True
                        self.best_validation_results[metric_name] = validation_results[metric_name]
                        metric_improved[metric_name] = True
                    results["validation_{}".format(metric_name)] = validation_results[metric_name].avg

                if train_epoch_info["last_step_of_epoch"] and self.save_epoch_freq == self.args["save_epoch_freq"] - 1:
                    self.save(
                        save_all=True,
                        is_best=one_metric_improved,
                        tags=best_select_metric if one_metric_improved else None,
                        identifier=next(self.counter),
                    )

                model_select_metric = self.args["model_select_metric"][0]

                if self.last_validation_metric is None:
                    self.last_validation_metric = validation_results[model_select_metric]
                else:
                    if validation_results[model_select_metric].avg > 0:
                        self.moving_average_metric_change = running_mean(
                            math.fabs(
                                (self.last_validation_metric.avg - validation_results[model_select_metric].avg)
                                / validation_results[model_select_metric].avg
                            ),
                            self.moving_average_metric_change,
                        )

                if "patience_metric_max_treshold"in self.args and self.args["patience_metric_max_treshold"]:
                    metric_exceeds_critical_treshold = validation_results[model_select_metric].avg_better_than_float(self.args["patience_metric_max_treshold"])
                else:
                    metric_exceeds_critical_treshold = False
                if "metric_not_achieving_critical_treshold"in self.args and self.args["metric_not_achieving_critical_treshold"]:
                    metric_not_achieving_critical_treshold = not validation_results[model_select_metric].avg_better_than_float(self.args["patience_metric_min_treshold"])
                else:
                    metric_not_achieving_critical_treshold = False
                if "patience_metric_change"in self.args and self.args["patience_metric_change"]:
                    metric_has_minimal_change = (
                            self.moving_average_metric_change is not None
                            and self.moving_average_metric_change < self.args["patience_metric_change"]
                    )
                else:
                    metric_has_minimal_change = False

                if (
                    metric_exceeds_critical_treshold
                    or metric_not_achieving_critical_treshold
                    or metric_has_minimal_change
                    or not metric_improved[model_select_metric]
                ):
                    logging.info(
                        "Loosing patience with {} in epoch {} because {}".format(
                            self.terminate_epochs,
                            model_select_metric,
                            " and ".join(
                                [
                                    a
                                    for a in [
                                        "metric_exceeds_critical_treshold"
                                        if metric_exceeds_critical_treshold
                                        else None,
                                        "metric_not_achieving_critical_treshold"
                                        if metric_not_achieving_critical_treshold
                                        else None,
                                        "metric_has_minimal_change" if metric_has_minimal_change else None,
                                        "metric has not improved" if not metric_improved[model_select_metric] else None,
                                    ]
                                    if a
                                ]
                            ),
                        )
                    )
                    if self.epoch >= self.terminate_epochs:
                        self.terminate = True
                else:
                    self.terminate_epochs = self.epoch + self.args["patience_epochs"]

                for optimizer in self.optimizers:
                    optimizer.lr_scheduler_step(validation_results[model_select_metric], epoch=self.epoch)

            self.results.add(**results)
            self.results.save()

    def load_state_dict(self, model: torch.nn.Module, state_dict, strict=True, freeze_param=False, weight_map=None):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """

        # own_state = dict(model.state_dict().items())
        own_state = model.state_dict(keep_vars=True)
        loaded_params = list()

        for name, param in state_dict.items():
            if weight_map is not None and name in weight_map:
                print("Mapping from {} to {}".format(name, weight_map[name]))
                name = weight_map[name]
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param_data = param.data
                else:
                    param_data = param
                try:
                    if isinstance(own_state[name], torch.nn.Parameter):
                        if param_data.size() != own_state[name].data.size():
                            if strict:
                                raise Exception
                            else:
                                continue
                        own_state[name].data.copy_(param_data)
                    else:
                        if param_data.size() != own_state[name].size():
                            if strict:
                                raise Exception
                            else:
                                continue
                        own_state[name].copy_(param_data)
                    if type(freeze_param) is list and (
                        name in freeze_param or type(freeze_param[0]) is str and freeze_param[0] == "True"
                    ):
                        logging.info("Freezing {}".format(name))
                        own_state[name].requires_grad = False
                except Exception as e:
                    print(e)
                    raise RuntimeError(
                        "While copying the parameter named {}, "
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}.".format(name, own_state[name].size(), param.size())
                    )
                loaded_params.append(name)
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
            else:
                print("{} missing from state_dict".format(name))
                pass
        missing = set(own_state.keys()) - set(state_dict.keys())
        if strict:
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        print('loaded from state_dict: "{}"'.format(loaded_params))
        # print('missing keys in state_dict: "{}"'.format(missing))

        for mod in model.modules():
            if hasattr(mod, "flatten_parameters"):
                mod.flatten_parameters()

    def load(
        self,
        filename,
        reset_optimizer=True,
        resume_filter=None,
        freeze_param=False,
        weight_map=None,
        checkpoint=None,
        dont_load_optimizer=False,
    ):
        if os.path.isfile(filename):
            if isinstance(self.args["devices"], tuple) or isinstance(self.args["devices"], list):
                main_device = self.args["devices"][0]
            else:
                main_device = self.args["devices"]
            if checkpoint is None:
                checkpoint = torch.load(
                    filename,
                    map_location={
                        "cuda:0": "cuda:{}".format(main_device),
                        "cuda:1": "cuda:{}".format(main_device),
                        "cuda:2": "cuda:{}".format(main_device),
                        "cuda:3": "cuda:{}".format(main_device),
                    },
                )
            state_dict = checkpoint["state_dict"]
            if resume_filter is not None:
                state_dict = {k: v for k, v in state_dict.items() if k in resume_filter}
            self.load_state_dict(self.model, state_dict, strict=False, freeze_param=freeze_param, weight_map=weight_map)
            self.training_steps = checkpoint["training_steps"]
            if "results" in checkpoint:
                self.results: ResultsLog = checkpoint["results"]
                self.results.set_path(os.path.join(self.save_path, "results.{}".format("csv")))
            if not dont_load_optimizer:
                for i, _ in enumerate(self.optimizers):
                    print("Loaded optimizer {}".format(i))
                    self.optimizers[i].load_state_dict(
                        checkpoint["optimizer_state_dict"][i],
                        self.epoch,
                        self.training_steps,
                        reset_optimizer=reset_optimizer,
                    )
            logging.info("loaded checkpoint '%s' (epoch %s)", filename, self.epoch)

        else:
            logging.error("invalid checkpoint: {}".format(filename))

    def save(self, filename=None, identifier=None, is_best=False, save_all=False, tags=None, keep_last=5):
        state = {
            "epoch": self.epoch,
            "training_steps": self.training_steps,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": [optim.state_dict() for optim in self.optimizers],
            "validation_results": getattr(self, "validation_results", None),
            "results": self.results,
        }
        state = dict(list(state.items()) + list(self.save_info.items()))
        identifier = identifier or ""
        filename = filename or self.checkpoint_filename % identifier
        filename = os.path.join(self.save_path, filename)
        logging.info("saving model to %s" % filename)
        torch.save(state, filename)
        if is_best:
            if tags is not None:
                for tag in tags:
                    mb_fn = "model_best-{}.pth.tar".format(tag)
                    if os.path.exists(os.path.join(self.save_path, mb_fn)):
                        mb_fn_epoch = "model_best-{}-{}.pth.tar".format(tag, identifier)
                        shutil.copyfile(os.path.join(self.save_path, mb_fn), os.path.join(self.save_path, mb_fn_epoch))
                    shutil.copyfile(filename, os.path.join(self.save_path, mb_fn))
            else:
                mb_fn = "model_best.pth.tar"
                if os.path.exists(os.path.join(self.save_path, mb_fn)):
                    mb_fn_epoch = "model_best-{}.pth.tar".format(identifier)
                    shutil.copyfile(os.path.join(self.save_path, mb_fn), os.path.join(self.save_path, mb_fn_epoch))
                shutil.copyfile(filename, os.path.join(self.save_path, "model_best.pth.tar"))
        if save_all:
            shutil.copyfile(filename, os.path.join(self.save_path, "checkpoint_epoch_%s.pth.tar" % self.epoch))
