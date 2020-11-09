import re

import torch.optim
import logging.config
from copy import deepcopy
from six import string_types

def eval_func(f, x):
    if isinstance(f, string_types):
        f = eval(f)
    return f(x)


class OptimRegime(object):
    """
    Reconfigures the optimizer according to setting list.
    Exposes optimizer methods - state, step, zero_grad, add_param_group

    Examples for regime:

        "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
          {'epoch': 2, 'optimizer': 'Adam', 'lr': 5e-4},
          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
          {'epoch': 8, 'optimizer': 'Adam', 'lr': 5e-5}
         ]"
    """

    def __init__(self, params, optimization_config, filter_weight_decay=False, lr_scheduler_config=None, ):
        self.optimizer = torch.optim.Adam(params, lr=0)
        if not isinstance(optimization_config, list):
            optimization_config['epoch'] = 0
            optimization_config = [optimization_config]
        if lr_scheduler_config is not None and not isinstance(lr_scheduler_config, list):
            lr_scheduler_config['epoch'] = 0
            lr_scheduler_config = [lr_scheduler_config]
        if filter_weight_decay:
            optimization_config = [{k:v for k, v in step.items() if k != 'weight_decay'} for step in optimization_config]
        self.optimization_config = optimization_config
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler = None
        # assert len(self.lr_scheduler_config) == len(self.optimization_config), "len(self.lr_scheduler_config) == len(self.optimization_config)"
        self.setting = {}
        self.current_optimization_config_phase = None

    @staticmethod
    def setup_optimizer_regime(args, model):
        optimizers = list()
        lr_scheduler_configs = (
            args["lr_scheduler_config"]
            if isinstance(args["lr_scheduler_config"], list) or isinstance(args["lr_scheduler_config"], tuple)
            else [args["lr_scheduler_config"]]
        ) if "lr_scheduler_config" in args else {}
        optimization_configs = (
            args["optimization_config"]
            if isinstance(args["optimization_config"], list) or isinstance(args["optimization_config"], tuple)
            else [args["optimization_config"]]
        )
        for optimization_config, lr_scheduler_config in zip(optimization_configs, lr_scheduler_configs):
            params = list()
            param_names = list()
            for name, p in OptimRegime.parameters(model):
                if not p.requires_grad:
                    continue
                if "match" in optimization_config:
                    if re.search(optimization_config["match"], name) is not None:
                        params.append(p)
                        param_names.append(name)
                else:
                    params.append(p)
                    param_names.append(name)
            if len(params) > 0:
                optimizers.append(
                    OptimRegime(
                        params, optimization_config=optimization_config, lr_scheduler_config=lr_scheduler_config
                    )
                )
            logging.info("PARAMS for optimzation config {}: ".format(optimization_config))
            for name in param_names:
                logging.info(name)
        return optimizers

    @staticmethod
    def parameters(parent_module, name_match=None, memo=None, prefix="",):
        for name, param in OptimRegime.named_parameters(parent_module, memo, prefix,):
            if name_match is not None:
                if name_match in name:
                    yield name, param
            else:
                yield name, param

    @staticmethod
    def named_parameters(parent_module, memo=None, prefix=""):
        if memo is None:
            memo = set()
        for name, p in parent_module._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ("." if prefix else "") + name, p
        for mname, module in parent_module.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            for name, p in OptimRegime.named_parameters(module, memo, submodule_prefix):
                yield name, p

    def update(self, epoch, train_steps, update_optimizer=False):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        if self.optimization_config is None:
            return
        if self.current_optimization_config_phase is None:
            update_optimizer = True
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_setting in enumerate(self.optimization_config):
                start_epoch = regime_setting.get('epoch', 0)
                start_step = regime_setting.get('step', 0)
                if epoch >= start_epoch or train_steps >= start_step:
                    self.current_optimization_config_phase = regime_phase
                    break
        if len(self.optimization_config) > self.current_optimization_config_phase + 1:
            next_phase = self.current_optimization_config_phase + 1
            # Any more regime steps?
            start_epoch = self.optimization_config[next_phase].get('epoch', float('inf'))
            start_step = self.optimization_config[next_phase].get('step', float('inf'))
            if epoch >= start_epoch or train_steps >= start_step:
                self.current_optimization_config_phase = next_phase
                update_optimizer = True

        optimizer_config = deepcopy(self.optimization_config[self.current_optimization_config_phase])

        lr_scheduler_config = None
        if self.lr_scheduler_config is not None:
            lr_scheduler_config = deepcopy(self.lr_scheduler_config[self.current_optimization_config_phase])

        if update_optimizer:
            self.adjust(optimizer_config, lr_scheduler_config)

    def get_current_setting(self):
        return deepcopy(self.optimization_config[self.current_optimization_config_phase])

    def adjust(self, optimizer_config: dict, lr_scheduler_config: dict):
        """adjusts optimizer according to a setting dict.
        e.g: setting={optimizer': 'Adam', 'lr': 5e-4}
        """
        if 'optimizer' in optimizer_config:
            optim_method = torch.optim.__dict__[optimizer_config['optimizer']]
            self.optimizer = optim_method(self.optimizer.param_groups)
            logging.info('OPTIMIZER - setting method = {}'.format(optimizer_config['optimizer']))
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in optimizer_config:
                    new_val = optimizer_config[key]
                    if new_val != param_group[key]:
                        logging.info('OPTIMIZER - setting {} = {}'.format(key, optimizer_config[key]))
                        param_group[key] = optimizer_config[key]
        if lr_scheduler_config is not None and 'lr_scheduler' in lr_scheduler_config:
            lr_scheduler_method = torch.optim.lr_scheduler.__dict__[lr_scheduler_config['lr_scheduler']]
            logging.info('LR SCHEDULER - setting method = {}'.format(lr_scheduler_config['lr_scheduler']))
            del lr_scheduler_config['lr_scheduler']
            del lr_scheduler_config['epoch']
            self.lr_scheduler = lr_scheduler_method(optimizer=self.optimizer, **lr_scheduler_config)
        self.setting = deepcopy(optimizer_config)

    def __getstate__(self):
        return {
            'optimizer_state': self.optimizer.__getstate__(),
            'regime': self.optimization_config,
        }

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        """
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'regime': self.optimization_config,
        }

    def load_state_dict(self, state_dict, epoch, train_steps, reset_optimizer=True):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # print(self.optimizer.param_groups)
        # print(state_dict['optimizer_state'])
        self.optimization_config = state_dict['regime']
        self.update(epoch, train_steps, update_optimizer=True)
        if not reset_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.optimizer.step(closure)

    def lr_scheduler_step(self, criterion_value, epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(criterion_value, epoch=epoch)

    def add_param_group(self, param_group):
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Variables should be optimized along with group
            specific optimization options.
        """
        self.optimizer.add_param_group(param_group)

