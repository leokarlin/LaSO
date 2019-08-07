from ignite.contrib.handlers.param_scheduler import ParamScheduler


class ManualParamScheduler(ParamScheduler):
    """A class for updating an optimizer's parameter value manually.

    Args:
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        param_name (str): name of optimizer's parameter to update
        param_callback (callable): A callback that should return the value.
        save_history (bool, optional): whether to log the parameter values
            (default=False)
    """
    def __init__(self, optimizer, param_name, param_callback, save_history=False):
        super(ManualParamScheduler, self).__init__(optimizer, param_name, save_history=save_history)

        self.param_callback = param_callback

    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        return self.param_callback()


