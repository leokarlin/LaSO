import time
import threading
from typing import Callable
from typing import Tuple

import py3nvml.py3nvml as nvml

def try_get_info(
    f : Callable[[int], object],
    h : object,
    default : object='N/A') -> int:
    """Safely try to call pynvml api."""

    try:
        v = f(h)
    except nvml.NVMLError_NotSupported:
        v = default

    return v


def gpu_info(gpu_index : int) -> Tuple[str, int]:
    """Returns a description of a GPU

    Returns the description and memory size of GPU.
    """

    nvml.nvmlInit()

    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    gpu_desc = nvml.nvmlDeviceGetName(handle)

    #
    # Get memory info.
    #
    mem_info = try_get_info(nvml.nvmlDeviceGetMemoryInfo, handle)
    if mem_info != 'N/A':
        mem_total = mem_info.total >> 20
    else:
        mem_total = 0

    return gpu_desc, mem_total


def query_gpu(index : int) -> Tuple[int, int, int]:

    h = nvml.nvmlDeviceGetHandleByIndex(index)

    #
    # Get memory info.
    #
    mem_info = try_get_info(nvml.nvmlDeviceGetMemoryInfo, h)
    if mem_info != 'N/A':
        mem_used = mem_info.used >> 20
        mem_total = mem_info.total >> 20
    else:
        mem_used = 0
        mem_total = 0

    #
    # Get utilization info
    #
    util = try_get_info(nvml.nvmlDeviceGetUtilizationRates, h)
    if util != 'N/A':
        gpu_util = util.gpu
    else:
        gpu_util = 0

    return mem_used, mem_total, gpu_util


class GPUMonitor(threading.Thread):
    shutdown = False
    daemon = True

    def __init__(
        self,
        gpu_index : int,
        callback : Callable[[int, int, int, int], None],
        sampling_period : float=0.5):
        """Utility class that monitors a specific GPU.

        Args:
            gpu_index (int): Index of GPU to monitor.
            callback (callback); Callback to call with GPU info.
            sampling_period (float): The monitoring period.
        """
        threading.Thread.__init__(self)
        self.gpu_index = gpu_index
        self.sampling_period = sampling_period
        self._monitor_callback = callback

    def run(self):

        #
        # Initialize nvml on the thread.
        #
        nvml.nvmlInit()

        t0 = time.time()
        while not self.shutdown:
            dt = int(time.time() - t0)

            mem_used, mem_total, gpu_util = query_gpu(self.gpu_index)
            self._monitor_callback(dt, mem_used, mem_total, gpu_util)

            time.sleep(self.sampling_period)

    def stop(self):
        self.shutdown = True
