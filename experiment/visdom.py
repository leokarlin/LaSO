import logging
import numpy as np
import os
import pprint
import threading
from typing import List, Tuple, Union
import visdom

_visdom = None
_save_by_default = None


def setup_visdom(
    server: str='',
    log_to_filename: str=None,
    save_by_default: bool=True) -> visdom.Visdom:
    """Setup the communication with the visdom server.

    Args:
        server (str): IP of visdom server.
        log_to_filename (str): If not None, a log of the plots will be saved to this path.  This log can be
            later replayed into visdom.
        save_by_default (bool): Automatically save the environment each update (Saved under '~/.visdom/<ENV>'
    """

    #
    # Setup visdom callbacks
    #
    global _visdom

    if server is None or not server:
        server = os.environ.get("VISDOM_SERVER_URL", 'http://localhost')

    username = os.environ.get("VISDOM_USERNAME", None)
    password = os.environ.get("VISDOM_PASSWORD", None)

    _visdom = visdom.Visdom(
        server=server,
        username=username,
        password=password,
        log_to_filename=log_to_filename,
    )

    if not _visdom.check_connection():
        raise RuntimeError("Visdom server not running. Please run " \
                           "python -m visdom.server")

    global _save_by_default

    _save_by_default = save_by_default

    return _visdom


class Window():
    """Class for creating visdom windows.

    Args:
        env (string): The visdom environment to log to.
        xlabel (string): xlabel of plot.
        ylabel (string): ylabel of plot.
        title (string): Title of the plot.
        showlegend (bool): Whether to show a legend.
    """

    def __init__(
        self,
        env,
        xlabel=None,
        ylabel=None,
        title=None,
        showlegend=False):

        assert _visdom is not None, "Need to first setup the communication with the visdom server."

        self._env = env
        self._win = None
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.showlegend = showlegend

        self._update = None

    @property
    def win(self):

        if self._win is None:
            #
            # Create an empty window.
            # TOOD
            # Avoid the hack of creating and removing a default trace.
            #
            self._win = _visdom.line(
                X=np.array([np.nan]),
                Y=np.array([np.nan]),
                env=self._env,
                name="default"
            )
            self._win = _visdom.line(
                X=np.array([np.nan]),
                Y=np.array([np.nan]),
                env=self._env,
                win=self._win,
                name="default",
                update="remove"
            )

        return self._win

    @property
    def update_mode(self):
        if self._update is None:
            self._update = "append"
            return None

        return self._update

    @property
    def env(self):
        return self._env

    @property
    def opts(self):
        return dict(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            showlegend=self.showlegend
        )


class Line():
    """Class that encapsulates the creation and use of line plots.

    Args:
        name (str) : Name of line (used in the legend). The name is also important
            when plotting multiple lines on the same plot.
        window (Window) : Window object to plot the line to.
        opts (dict) : Optional visdom `opts` parameters to apply to the line.
    """

    def __init__(self, name: str, window: Window, opts: dict=None):
        self.name = name
        self._window = window
        self._line_opts = opts
        self._last_opts = {}

    def append(self, x: float, y: float, opts: dict=None):

        _visdom.line(
            X=np.array([x]),
            Y=np.array([y]),
            env=self._window.env,
            win=self._window.win,
            name=self.name,
            update=self._window.update_mode,
        )

        new_opts = self._window.opts
        if self._line_opts is not None:
            new_opts.update(self._line_opts)
        if opts is not None:
            new_opts.update(opts)

        if new_opts != self._last_opts:
            #
            # Need to update the opts.
            #
            _visdom.update_window_opts(
                win=self._window.win,
                opts=new_opts,
                env=self._window.env
            )

            self._last_opts = new_opts

        if _save_by_default:
            _visdom.save([self._window.env])


class VisdomLogHandler(logging.Handler):
    """Logging handler that logs to a Visdom text window.

    Args:
        env (string): The visdom environment to log to.
        title (string): Title of the logger window
    """

    def __init__(self, env, title="Logging", *args, **kwds):

        super().__init__(*args, **kwds)

        self.env = env
        self.win = _visdom.text("<h5>{}</h5>".format(title), env=env)

    def emit(self, record):
        log_entry = self.format(record)

        _visdom.text(win=self.win, text=log_entry, env=self.env, append=True)


def create_plot_window(vis, env, xlabel, ylabel, title, legend=None):
    """Create a plot window.

    Args:
        vis (object): The Visdom object.
        env (string): The visdom environment to log to.
        xlabel (string): xlabel of plot.
        ylabel (string): ylabel of plot.
        title (string): Title of the plot.
        legend (string): Legend of line.

    Note:
        Depracated. Use the new Window interface.
    """
    win = vis.line(
        X=np.array([1]),
        Y=np.array([np.nan]),
        opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend),
        env=env
    )
    return win


def write_conf(env, args=None, text=None):
    """Write configuration to the Visdom env.

    Args:
        env (string): The visdom environment to log to.
        args (Namespace, optional): The argument namespace returned by argparse.
        text (string, optional): Configuration as text block.
    """

    conf_text = ""
    if args:
        conf_text += pprint.pformat(args.__dict__, indent=4)
    if text:
        conf_text += text
    conf_text = "<br />".join(conf_text.split("\n"))

    win = _visdom.text("<h5>Configuration</h5>", env=env)
    _visdom.text(win=win, env=env,
             text="<p>{}</p>".format(conf_text),
             append=True)


def monitor_gpu(env : str, gpu_index : int=None, xtick_size : int=100) -> threading.Thread:
    """Monitor the memory and utilization of a GPU.

    Args:
        env (str): The visdom environment to log to.
        gpu_index (int): The GPU to monitor.
    """

    import CCC.monitor as mon

    if gpu_index is None:
        gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"])

    desc, total = mon.gpu_info(gpu_index)
    title = "{} ({}G)".format(desc, total >> 10)
    win = Window(env=env, xlabel="time [s]", ylabel="percent [%]", title=title)

    util_plot = Line("util", win, opts=dict(ytickmin=0, ytickmax=100))
    mem_plot = Line("mem", win, opts=dict(ytickmin=0, ytickmax=100))

    def cb(dt, mem_used, mem_total, gpu_util):
        xtickmin = max(0, dt-xtick_size)
        xtickmax = max(xtick_size, dt)

        mem_plot.append(x=dt, y=int(mem_used / total * 100), opts=dict(xtickmin=xtickmin, xtickmax=xtickmax))
        util_plot.append(x=dt, y=gpu_util, opts=dict(xtickmin=xtickmin, xtickmax=xtickmax))

    sm = mon.GPUMonitor(gpu_index, cb)
    sm.start()

    return sm


class ParametersControlWindow(Window):
    """Visdom window for controlling parameters.

    Args:
        env (string): The visdom environment to log to.
    """

    def _form_properties(self):
        properties = []
        for param in self.parameters:
            properties.append(
                {'type': 'number', 'name': param.name, 'value': str(param.value)}
            )
        return properties

    def _property_updated(self, new_value: float):
        _visdom.properties(self._form_properties(), win=self._win, env=self._env)

    def _properties_callback(self, event):
        if event['event_type'] == 'PropertyUpdate':
            prop_id = event['propertyId']
            new_value = event['value']
            self.parameters[prop_id].value = new_value

    def register_parameters(self, parameters: Union[List, Tuple]):
        """Register parameters for controlling in the window.

        Args:
            parameters (list): List of Parameters to control.
        """
        self.parameters = parameters
        for param in self.parameters:
            param._callbacks.append(self._property_updated)

        self._win = _visdom.properties(self._form_properties(), env=self._env)
        _visdom.register_event_handler(
            self._properties_callback,
            self._win
        )


class ParametersViewWindow(Window):
    """Visdom window for viewing parameters.

    Args:
        env (string): The visdom environment to log to.
        xlabel (string): xlabel of plot.
        ylabel (string): ylabel of plot.
        title (string): Title of the plot.
        showlegend (bool): Whether to show a legend.
    """
    def register_parameters(self, parameters: Union[List, Tuple]):
        """Register parameters for viewing in the window.

        Args:
            parameters (list): List of Parameter objects to view.
        """
        self.parameters = parameters
        self.parameters_lines = []
        for param in self.parameters:
            self.parameters_lines.append(Line(param.name, window=self))

    def update(self, x: float):
        """Update the values of the parameters on the view.

        Args:
            x (float): x value of graph.
        """
        for param, param_line in zip(self.parameters, self.parameters_lines):
            param_line.append(
                x=x,
                y=param.value
            )


def create_parameters_windows(
    parameters: Union[List, Tuple],
    env: str,
    xlabel: str="iteration",
    ylabel: str="value",
    title: str="Manual Parameters",
    showlegend: bool=False):
    """Create control and view windows for parameters

    Args:
        env (string): The visdom environment to log to.
        xlabel (string): xlabel of plot.
        ylabel (string): ylabel of plot.
        title (string): Title of the plot.
        showlegend (bool): Whether to show a legend.
    """
    #
    # Create the properties control window
    #
    params_control_win = ParametersControlWindow(env)
    params_control_win.register_parameters(parameters)

    params_view_win = ParametersViewWindow(
        env, xlabel=xlabel, ylabel=ylabel, title=title, showlegend=showlegend
    )
    params_view_win.register_parameters(parameters)

    return params_control_win, params_view_win
