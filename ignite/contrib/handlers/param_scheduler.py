from __future__ import division

from copy import deepcopy

import math

from abc import ABCMeta, abstractmethod
from ignite._six import with_metaclass

import torch
from torch.optim.lr_scheduler import _LRScheduler


class ParamScheduler(with_metaclass(ABCMeta, object)):
    """An abstract class for updating an optimizer's parameter value during
    training.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use.
        param_name (str): name of optimizer's parameter to update.
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).
    """

    def __init__(self, optimizer, param_name, save_history=False):

        if isinstance(optimizer, dict):
            self.optimizer_param_groups = [optimizer]
        else:
            self.optimizer_param_groups = optimizer.param_groups
        self.param_name = param_name
        self.save_history = save_history
        self.event_index = 0

    def __call__(self, engine, name=None):

        value = self.get_param()

        for param_group in self.optimizer_param_groups:
            param_group[self.param_name] = value

        if name is None:
            name = self.param_name

        if self.save_history:
            if not hasattr(engine.state, 'param_history'):
                setattr(engine.state, 'param_history', {})
            engine.state.param_history.setdefault(name, [])
            values = [pg[self.param_name] for pg in self.optimizer_param_groups]
            engine.state.param_history[name].append(values)

        self.event_index += 1

    @abstractmethod
    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        pass

    @classmethod
    def simulate_values(cls, num_events, **scheduler_kwargs):
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events (int): number of events during the simulation.
            **scheduler_kwargs : parameter scheduler configuration kwargs.

        Returns:
            list of pairs: [event_index, value]

        Examples:

        .. code-block:: python

            lr_values = np.array(LinearCyclicalScheduler.simulate_values(num_events=50, param_name='lr',
                                                                         start_value=1e-1, end_value=1e-3,
                                                                         cycle_size=10))

            plt.plot(lr_values[:, 0], lr_values[:, 1], label="learning rate")
            plt.xlabel("events")
            plt.ylabel("values")
            plt.legend()

        """
        keys_to_remove = ['optimizer', 'save_history']
        for key in keys_to_remove:
            if key in scheduler_kwargs:
                del scheduler_kwargs[key]
        values = []
        scheduler = cls(optimizer={}, save_history=False, **scheduler_kwargs)
        for i in range(num_events):
            scheduler(engine=None)
            values.append([i, scheduler.optimizer_param_groups[0][scheduler.param_name]])
        return values


class CyclicalScheduler(ParamScheduler):
    """An abstract class for updating an optimizer's parameter value over a
    cycle of some size.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use.
        param_name (str): name of optimizer's parameter to update.
        start_value (float): value at start of cycle.
        end_value (float): value at the middle of the cycle.
        cycle_size (int): length of cycle.
        cycle_mult (float, optional): ratio by which to change the cycle_size.
            at the end of each cycle (default=1.0).
        start_value_mult (float, optional): ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult (float, optional): ratio by which to change the end value at the
            end of each cycle (default=1.0).
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.
    """

    def __init__(self,
                 optimizer,
                 param_name,
                 start_value,
                 end_value,
                 cycle_size,
                 cycle_mult=1.0,
                 start_value_mult=1.0,
                 end_value_mult=1.0,
                 save_history=False):
        super(CyclicalScheduler, self).__init__(
            optimizer,
            param_name,
            save_history=save_history
        )
        self.start_value = start_value
        self.end_value = end_value
        self.cycle_size = cycle_size
        self.cycle_mult = cycle_mult
        self.cycle = 0
        self.start_value_mult = start_value_mult
        self.end_value_mult = end_value_mult

    def __call__(self, engine, name=None):
        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1
            self.start_value *= self.start_value_mult
            self.end_value *= self.end_value_mult

        return super(CyclicalScheduler, self).__call__(engine, name)


class LinearCyclicalScheduler(CyclicalScheduler):
    """Linearly adjusts param value to 'end_value' for a half-cycle, then linearly
    adjusts it back to 'start_value' for a half-cycle.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use.
        param_name (str): name of optimizer's parameter to update.
        start_value (float): value at start of cycle.
        end_value (float): value at the middle of the cycle.
        cycle_size (int): length of cycle.
        cycle_mult (float, optional): ratio by which to change the cycle_size
            at the end of each cycle (default=1).
        start_value_mult (float, optional): ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult (float, optional): ratio by which to change the end value at the
            end of each cycle (default=1.0).
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

        scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1e-3, 1e-1, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        #
        # Linearly increases the learning rate from 1e-3 to 1e-1 and back to 1e-3
        # over the course of 1 epoch
        #
    """

    def get_param(self):
        cycle_progress = self.event_index / self.cycle_size
        return self.end_value + (self.start_value - self.end_value) * abs(cycle_progress - 0.5) * 2


class CosineAnnealingScheduler(CyclicalScheduler):
    """Anneals 'start_value' to 'end_value' over each cycle.

    The annealing takes the form of the first half of a cosine
    wave (as suggested in [Smith17]_).

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use.
        param_name (str): name of optimizer's parameter to update.
        start_value (float): value at start of cycle.
        end_value (float): value at the end of the cycle.
        cycle_size (int): length of cycle.
        cycle_mult (float, optional): ratio by which to change the cycle_size
            at the end of each cycle (default=1).
        start_value_mult (float, optional): ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult (float, optional): ratio by which to change the end value at the
            end of each cycle (default=1.0).
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

        scheduler = CosineAnnealingScheduler(optimizer, 'lr', 1e-1, 1e-3, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        #
        # Anneals the learning rate from 1e-1 to 1e-3 over the course of 1 epoch.
        #

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

        optimizer = SGD(
            [
                {"params": model.base.parameters(), 'lr': 0.001),
                {"params": model.fc.parameters(), 'lr': 0.01),
            ]
        )

        scheduler1 = LinearCyclicalScheduler(optimizer.param_groups[0], 'lr', 1e-7, 1e-5, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler1, "lr (base)")

        scheduler2 = CosineAnnealingScheduler(optimizer.param_groups[1], 'lr', 1e-5, 1e-3, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler2, "lr (fc)")

    .. [Smith17] Smith, Leslie N. "Cyclical learning rates for training neural networks."
                 Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017
    """

    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        cycle_progress = self.event_index / self.cycle_size
        return self.start_value + ((self.end_value - self.start_value) / 2) * (1 - math.cos(math.pi * cycle_progress))


class ConcatScheduler(ParamScheduler):
    """Concat a list of parameter schedulers.

    The `ConcatScheduler` goes through a list of schedulers given by `schedulers`. Duration of each
    scheduler is defined by `durations` list of integers.

    Args:
        schedulers (list of ParamScheduler): list of parameter schedulers.
        durations (list of int): list of number of events that lasts a parameter scheduler from schedulers.
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import ConcatScheduler
        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

        scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.1, end_value=0.5, cycle_size=60)
        scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.5, end_value=0.01, cycle_size=60)

        combined_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[30, ])
        trainer.add_event_handler(Events.ITERATION_COMPLETED, combined_scheduler)
        #
        # Sets the Learning rate linearly from 0.1 to 0.5 over 30 iterations. Then
        # starts an annealing schedule from 0.5 to 0.01 over 60 iterations.
        # The annealing cycles are repeated indefinitely.
        #
    """

    def __init__(self, schedulers, durations, save_history=False):

        if not isinstance(schedulers, (list, tuple)) or len(schedulers) < 2:
            raise ValueError("Argument schedulers should be list/tuple of more than one parameter schedulers, "
                             "but given {}".format(schedulers))

        if not isinstance(durations, (list, tuple)) or sorted(list(durations)) != list(durations):
            raise ValueError("Argument durations should be list/tuple of ordered integers, "
                             "but given {}".format(durations))

        if len(schedulers) != len(durations) + 1:
            raise ValueError("Incorrect number schedulers or duration values")

        for i, scheduler in enumerate(schedulers):
            if not isinstance(scheduler, ParamScheduler):
                raise TypeError("Value at index {} of schedulers should be a parameter scheduler, "
                                "but given {}".format(i, type(scheduler)))

        super(ConcatScheduler, self).__init__(optimizer={}, param_name="", save_history=save_history)
        self.schedulers = list(schedulers)
        self.durations = list(durations) + [-1, ]
        self._current_scheduler = None
        self._current_duration = None
        self._set_next_scheduler()

    def _set_next_scheduler(self):
        self._current_scheduler = self.schedulers.pop(0)
        self._current_scheduler.save_history = self.save_history
        self._current_duration = self.durations.pop(0)
        self.optimizer_param_groups = self._current_scheduler.optimizer_param_groups
        self.param_name = self._current_scheduler.param_name

    def __call__(self, engine, name=None):
        if self._current_duration == 0:
            self._set_next_scheduler()

        self._current_scheduler(engine, name)
        self._current_duration -= 1

    def get_param(self):
        pass

    @classmethod
    def simulate_values(cls, num_events, schedulers, durations, param_names=None, **kwargs):
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events (int): number of events during the simulation.
            schedulers (list of ParamScheduler): list of parameter schedulers.
            durations (list of int): list of number of events that lasts a parameter scheduler from schedulers.
            param_names (list or tuple of str, optional): parameter name or list of parameter names to simulate values.
                By default, the first scheduler's parameter name is taken.

        Returns:
            list of [event_index, value_0, value_1, ...], where values correspond to `param_names`.

       """
        if param_names is not None and not isinstance(param_names, (tuple, list)):
            raise ValueError("Argument param_names should be list or tuple of strings")
        output = []

        # Need to copy schedulers otherwise unsafe
        copy_schedulers = []
        for s in schedulers:
            if isinstance(s, LRScheduler):
                s = LRScheduler(LRScheduler._copy_lr_scheduler(s.lr_scheduler))
            else:
                s = deepcopy(s)
            copy_schedulers.append(s)

        scheduler = cls(copy_schedulers, durations)
        if param_names is None:
            param_names = [scheduler.param_name]
        for i in range(num_events):
            scheduler(engine=None)
            values = [scheduler.optimizer_param_groups[0][param_name] for param_name in param_names]
            output.append([i, ] + values)
        return output


class LRScheduler(ParamScheduler):
    """A wrapper class to call `torch.optim.lr_scheduler` objects as `ignite` handlers.

    Args:
        lr_scheduler (subclass of `torch.optim.lr_scheduler._LRScheduler`): lr_scheduler object to wrap.
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import LRScheduler
        from torch.optim.lr_scheduler import StepLR

        step_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        scheduler = LRScheduler(step_scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    """

    def __init__(self, lr_scheduler, save_history=False, **kwds):

        if not isinstance(lr_scheduler, _LRScheduler):
            raise TypeError("Argument lr_scheduler should be a subclass of torch.optim.lr_scheduler._LRScheduler, "
                            "but given {}".format(type(lr_scheduler)))

        if len(lr_scheduler.optimizer.param_groups) > 1:
            raise ValueError("Optimizer passed to lr_scheduler should have a single param group, "
                             "but currently there are {} param groups".format(len(lr_scheduler.optimizer.param_groups)))

        self.lr_scheduler = lr_scheduler
        super(LRScheduler, self).__init__(
            optimizer=self.lr_scheduler.optimizer,
            param_name='lr',
            save_history=save_history
        )

    def __call__(self, engine, name=None):
        self.lr_scheduler.last_epoch += 1
        super(LRScheduler, self).__call__(engine, name)

    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        lr_list = self.lr_scheduler.get_lr()
        if len(lr_list) > 1:
            raise ValueError("Optimizer passed to lr_scheduler should have a single param group, "
                             "but currently there are {} param groups".format(len(lr_list)))
        return lr_list[0]

    @classmethod
    def simulate_values(cls, num_events, lr_scheduler, **kwargs):
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events (int): number of events during the simulation.
            lr_scheduler (subclass of `torch.optim.lr_scheduler._LRScheduler`): lr_scheduler object to wrap.

        Returns:
            list of pairs: [event_index, value]

        """
        copy_lr_scheduler = LRScheduler._copy_lr_scheduler(lr_scheduler)
        values = []
        scheduler = cls(save_history=False, lr_scheduler=copy_lr_scheduler)
        for i in range(num_events):
            scheduler(engine=None)
            values.append([i, scheduler.optimizer_param_groups[0][scheduler.param_name]])

        return values

    @staticmethod
    def _copy_lr_scheduler(lr_scheduler):
        lr_scheduler_cls = lr_scheduler.__class__
        optimizer_cls = lr_scheduler.optimizer.__class__
        t = torch.zeros([1], requires_grad=True)
        dummy_optimizer = optimizer_cls([t], lr=0.1)
        dummy_optimizer.load_state_dict(lr_scheduler.optimizer.state_dict())
        kwargs = lr_scheduler.state_dict()
        del kwargs['base_lrs']
        copy_lr_scheduler = lr_scheduler_cls(optimizer=dummy_optimizer, **kwargs)
        copy_lr_scheduler.load_state_dict(lr_scheduler.state_dict())
        return copy_lr_scheduler


def create_lr_scheduler_with_warmup(lr_scheduler, warmup_start_value, warmup_end_value, warmup_duration,
                                    save_history=False,
                                    output_simulated_values=None):
    """
    Helper method to create a LR scheduler with a linear warm-up.

    Args:
        lr_scheduler (ParamScheduler or subclass of `torch.optim.lr_scheduler._LRScheduler`): LR scheduler after
            the warm-up.
        warmup_start_value (float): LR start value of the warm-up phase.
        warmup_end_value (float): LR end value of the warm-up phase.
        warmup_duration (int): warm-up phase duration, number of events.
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        output_simulated_values (list or tuple, optional): optional output of simulated LR values.
            If output_simulated_values is set to an empty list, after the execution it will be filled
            by simulated LR values.

    Returns:
        ConcatScheduler: LR scheduler with linear warm-up.


    .. code-block:: python

        torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)
        lr_values = []
        scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                    warmup_start_value=0.0, warmup_end_value=0.1, warmup_duration=10,
                                                    output_simulated_values=lr_values)
        lr_values = np.array(lr_values)
        # Plot simulated values
        plt.plot(lr_values[:, 0], lr_values[:, 1], label="learning rate")

        # Attach to the trainer
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    """
    if not isinstance(lr_scheduler, (ParamScheduler, _LRScheduler)):
        raise TypeError("Argument lr_scheduler should be a subclass of torch.optim.lr_scheduler._LRScheduler or "
                        "ParamScheduler, but given {}".format(type(lr_scheduler)))

    if isinstance(lr_scheduler, _LRScheduler):
        lr_scheduler = LRScheduler(lr_scheduler)

    dummy_optimizer = {}
    warmup_scheduler = LinearCyclicalScheduler(dummy_optimizer, param_name="lr",
                                               start_value=warmup_start_value,
                                               end_value=warmup_end_value,
                                               cycle_size=warmup_duration * 2)

    warmup_scheduler.optimizer_param_groups = lr_scheduler.optimizer_param_groups

    schedulers = [warmup_scheduler, lr_scheduler]
    durations = [warmup_duration, ]
    combined_scheduler = ConcatScheduler(schedulers, durations=durations,
                                         save_history=save_history)
    if output_simulated_values is not None:
        output_simulated_values.extend(ConcatScheduler.simulate_values(num_events=warmup_duration * 20,
                                                                       schedulers=schedulers, durations=durations))
    return combined_scheduler


class ReduceLROnPlateau(ParamScheduler):
    """An adapter class to call `torch.optim.lr_scheduler.ReduceLROnPlateau` object as `ignite` handlers.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        metric_name (str, optional): name of  metrics by which to
             measure plateau (either `metric_name` or
            `output_transform` should be used, but not both).
        output_transform (Callable, optional): a function to select
            (from the engine's output) what you want
            to measure plateau by. This function should return a
            single scalar. (either `metric_name` or
            `output_transform` should be used, but not both).
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        save_history (bool, optional): whether to log the parameter values
            (default=False)

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(optimizer, output_transform=lambda x: x, patience=2, factor=0.2)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    """
    def __init__(self, optimizer, metric_name=None, output_transform=None, mode='min',
                 factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8, save_history=False, **kwds):

        assert (metric_name is None and output_transform is not None) or \
               (metric_name is not None and output_transform is None), \
            "One of the parameters: `metric_name`, `output_transform` should be used but not both. "

        from torch.optim.lr_scheduler import ReduceLROnPlateau as pytorch_ROP

        self._lr_scheduler = pytorch_ROP(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=False,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
        )
        self.metric_name = metric_name
        self.output_transform = output_transform

        super(ReduceLROnPlateau, self).__init__(
            optimizer=optimizer,
            param_name='lr',
            save_history=save_history
        )

    def __call__(self, engine, name=None):

        #
        # Call the pytorch scheduler step method.
        #
        if self.metric_name is not None:
            if self.metric_name not in engine.state.metrics:
                raise KeyError("metric {} not found in engine.state.metrics".format(self.metric_name))

            metric = engine.state.metrics[self.metric_name]
        else:
            metric = self.output_transform(engine.state.output)

        self._lr_scheduler.step(metric, engine.state.epoch)

        #
        # Call the ignite object __call__ method.
        # Note:
        # The 'lr' parameter will be updated for the second time but with the same value.
        #
        super(ReduceLROnPlateau, self).__call__(engine, name)

    def get_param(self):
        """Method to get current optimizer's parameter value
        """

        #
        # The current optimizer lr is read from ... the current optimizer lr.
        #
        param_group = list(self.optimizer_param_groups)[0]

        return param_group[self.param_name]
