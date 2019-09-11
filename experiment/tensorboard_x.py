from functools import partial
import logging
import numpy as np
import os
import pprint
import threading
from typing import Any, Union, List, Tuple, Dict

from tensorboardX import SummaryWriter


class TensorBoardXLogHandler(logging.Handler):
    """Logging handler that logs to a TensorBoardX instance.

    Args:
        summary_writer (tensorboard.SummaryWriter): The summarywriter to log to.
        title (string): Title/tag to write to.
    """

    def __init__(
        self,
        summary_writer,   # type: SummaryWriter
        title="Logging",  # type: str
        *args, **kwds
    ):

        super(TensorBoardXLogHandler, self).__init__(*args, **kwds)

        self.summary_writer = summary_writer
        self.title = title
        self.global_step = 0
        self.accomulated_entries = ""

    def emit(self, record):
        log_entry = self.format(record)

        #
        # There seems to be a bug in writing new lines:
        # https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
        #
        self.accomulated_entries += "  \n"
        self.accomulated_entries += log_entry.replace("\n", "  \n")

        self.summary_writer.add_text(
            tag=self.title,
            text_string=self.accomulated_entries,
            global_step=self.global_step
        )


def write_conf(
    summary_writer,  # type: SummaryWriter
    args=None, text=None):
    """Write configuration to the Visdom env.

    Args:
        summary_writer (tensorboard.SummaryWriter): The summarywriter to log to.
        args (Namespace, optional): The argument namespace returned by argparse.
        text (string, optional): Configuration as text block.
    """

    conf_text = ""
    if args:
        conf_text += pprint.pformat(args.__dict__, indent=4)
    if text:
        conf_text += text

    #
    # There seems to be a bug in writing new lines:
    # https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
    #
    conf_text = conf_text.replace("\n", "  \n")

    summary_writer.add_text(
        tag="Configuration",
        text_string=conf_text
    )


def monitor_gpu(
    summary_writer,  # type: SummaryWriter
    gpu_index=None,  # type: int
    xtick_size=100,  # type: int
): # -> threading.Thread
    """Monitor the memory and utilization of a GPU.

    Args:
        env (str): The visdom environment to log to.
        gpu_index (int): The GPU to monitor.
    """

    import CCC.monitor as mon

    if gpu_index is None:
        gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"])

    desc, total = mon.gpu_info(gpu_index)
    title = desc.replace(' ', '_')

    def cb(dt, mem_used, mem_total, gpu_util):

        summary_writer.add_scalar(
            tag='/'.join([title, "mem"]),
            scalar_value=int(mem_used / total * 100),
            global_step=dt
        )

        summary_writer.add_scalar(
            tag='/'.join([title, "util"]),
            scalar_value=gpu_util,
            global_step=dt
        )

    sm = mon.GPUMonitor(gpu_index, cb)
    sm.start()

    return sm

