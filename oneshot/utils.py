import os
class conditional(object):
    """Wrap another context manager and enter it only if condition is true.
    """

    def __init__(self, condition, contextmanager):
        self.condition = condition
        self.contextmanager = contextmanager

    def __enter__(self):
        if self.condition:
            return self.contextmanager.__enter__()

    def __exit__(self, *args):
        if self.condition:
            return self.contextmanager.__exit__(*args)


def setupCUDAdevice(cuda_visible_device=None):
    """Setup `CUDA_VISIBLE_DEVICES` environment variable.

    The `CUDA_VISIBLE_DEVICES` environment variable is used for determining the
    GPU available to a process. It is automatically set by the `jbsub` command.
    In some situations a user would like to connect to a GPU node not through
    a job, e.g. using a remote debugger. In this cases the `CUDA_VISIBLE_DEVICES`
    will not be set, even if the user has secured a GPU through a separate
    interactive session. `setupCUDAdevice` can be used to query the user for the
    available device number. It does nothing when `CUDA_VISIBLE_DEVICES` is
    already set.

    Args:
        cuda_visible_device (int): Hard code a device number (will by pass
        user input).

    .. note::
        Using hard coded `cuda_visible_device` should be used carefully.
    """

    if "CUDA_VISIBLE_DEVICES" not in os.environ or \
        os.environ["CUDA_VISIBLE_DEVICES"] == "":
        if cuda_visible_device is None:
            cuda_visible_device = input("CUDA_VISIBLE_DEVICES: ")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
