import numpy as np
from functools import reduce
import cupy
import chainer
from chainer.training import extension


# The name template of the statistic to collect and include in the report,
# e.g. 'predictor/conv1/W/grad/percentile/sigma_one'
key_template = '{model}/{layer}/{param}/{attr}/{statistic}'


def weight_statistics(model, layer_name=None):

    """Collect weight statistict from the given model and return it as a
    ``dict``.

    Args:
        model (~chainer.Chain): The model from which statistics are collected.
        layer_name (str): Name of the layer which may be specified or set to
            ``None`` to aggregate over all layers.

    Returns:
        dict: Parameter statistics.
    """

    return parameter_statistics(model, 'W', 'data', layer_name)


def bias_statistics(model, layer_name=None):

    """Collect bias statistict from the given model and return it as a
    ``dict``.

    Args:
        model (~chainer.Chain): The model from which statistics are collected.
        layer_name (str): Name of the layer which may be specified or set to
            ``None`` to aggregate over all layers.

    Returns:
        dict: Parameter statistics.
    """

    return parameter_statistics(model, 'b', 'data', layer_name)


def weight_gradient_statistics(model, layer_name=None):

    """Collect weight gradient statistict from the given model and return it
    as a ``dict``.

    Args:
        model (~chainer.Chain): The model from which statistics are collected.
        layer_name (str): Name of the layer which may be specified or set to
            ``None`` to aggregate over all layers.

    Returns:
        dict: Parameter statistics.
    """

    return parameter_statistics(model, 'W', 'grad', layer_name)


def bias_gradient_statistics(model, layer_name=None):

    """Collect bias gradient statistict from the given model and return it
    as a ``dict``.

    Args:
        model (~chainer.Chain): The model from which statistics are collected.
        layer_name (str): Name of the layer which may be specified or set to
            ``None`` to aggregate over all layers.

    Returns:
        dict: Parameter statistics.
    """

    return parameter_statistics(model, 'b', 'grad', layer_name)


def sparsity(model, include_bias=False, layer_name=None):

    """Count the number of parameters with the value zero for the given model
    and return it as a ``dict``.

    Args:
        model (~chainer.Chain): The model from which statistics are collected.
        include_bias (bool): ``True`` to include the number of biases that are
            zero, ``False`` to exclude them.
        layer_name (str): Name of the layer which may be specified or set to
            ``None`` to aggregate over all layers.

    Returns:
        dict: Parameter statistics.
    """

    xp = model.xp

    def reduce_count_zeros(acc, param):
        if param.name == 'W' or (include_bias and param.name == 'b'):
            acc += param.data.size - xp.count_nonzero(param.data)
        return acc

    if layer_name is not None:
        sparsity = reduce(reduce_count_zeros, [getattr(model, layer_name)], 0)
    else:
        sparsity = reduce(reduce_count_zeros, model.params(), 0)

    key = key_template.format(model=model.name,
                              layer='*' if layer_name is None else layer_name,
                              param='Wb' if include_bias else 'W' ,
                              attr='sparsity',
                              statistic='zeros')

    return { key: sparsity }


def parameter_statistics(model, param_name, attr_name, layer_name=None):

    """Collect statistict from the given model and return it as a ``dict``.

    The returned ``dict`` contains a key for each metric, mapping to a NumPy
    or CuPy ``float32`` value depending on if the given model was on the CPU or
    the GPU.

    Args:
        model (~chainer.Chain): The model from which statistics are collected.
        param_name (str): Name of the parameter, ``'W'`` or ``'b'``.
        attr_name (str): Name of the attribute, ``'data'`` or ``'grad'``.
        layer_name (str): Name of the layer which may be specified or set to
            ``None`` to aggregate over all layers.

    Returns:
        dict: Parameter statistics.
    """

    if layer_name is not None:  # Collect statistics for a single layer only
        l = getattr(model, layer_name)
        lp = layer_params(l, param_name, attr_name)
        return as_statistics(lp, model.name, param_name, attr_name,
                             layer_name=layer_name)

    lp = layers_params(model, param_name, attr_name)
    return as_statistics(lp, model.name, param_name, attr_name)


def layer_params(layer, param_name, attr_name):

    """Return parameters in a flattened array from the given layer or an empty
    array if the parameters are not found.

    Args:
        layer (~chainer.Link): The layer from which parameters are collected.
        param_name (str): Name of the parameter, ``'W'`` or ``'b'``.
        attr_name (str): Name of the attribute, ``'data'`` or ``'grad'``.

    Returns:
        array: Flattened array of parameters.
    """

    if isinstance(layer, chainer.Chain):
        # Nested chainer.Chain, aggregate all underlying statistics
        return layers_params(layer, param_name, attr_name)
    elif not hasattr(layer, param_name):
        return layer.xp.array([])

    params = getattr(layer, param_name)
    params = getattr(params, attr_name)
    return params.flatten()


def layers_params(model, param_name, attr_name):

    """Return all parameters in a flattened array from the given model.

    Args:
        model (~chainer.Chain): The model from which parameters are collected.
        param_name (str): Name of the parameter, ``'W'`` or ``'b'``.
        attr_name (str): Name of the attribute, ``'data'`` or ``'grad'``.

    Returns:
        array: Flattened array of parameters.
    """

    xp = model.xp
    params = xp.array([], dtype=xp.float32)

    for param in model.params():
        if param.name == param_name:
            values = getattr(param, attr_name)
            values = values.flatten()
            params = xp.concatenate((params, values))  # Slow?

    return params


def as_statistics(data, model_name, param_name, attr_name, *, layer_name=None,
                  statistics=('min', 'max', 'mean', 'std'),
                  percentiles=(0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87)):

    """Compute statistics based on the given data and return it as a ``dict``.

    Args:
        data (array): NumPy or CuPy array of data.
        model_name (str): Name of the model,  e.g. ``predictor``.
        param_name (str): Name of the parameter, ``'W'`` or ``'b'``.
        attr_name (str): Name of the attribute, ``'data'`` or ``'grad'``.
        layer_name (str): Name of the layer which may be specified or set to
            ``None``. In the case of ``None`` the layer name will be set to
            ``'*'``.

    Returns:
        dict: Parameter statistics.
    """

    stats = {}

    if layer_name is None:
        layer_name = '*'

    if percentiles:
        ps = get_percentiles(data, sigma=percentiles)
        for i, percentile in enumerate(ps):
            key = key_template.format(model=model_name,
                                      layer=layer_name,
                                      param=param_name,
                                      attr=attr_name,
                                      statistic='percentile/{}'.format(i))
            stats[key] = percentile

    for s in statistics:
        key = key_template.format(model=model_name,
                                  layer=layer_name,
                                  param=param_name,
                                  attr=attr_name,
                                  statistic=s)
        try:
            stats[key] = getattr(data, s)()
        except ValueError:
            # If data is missing from uninitialized model parameters, add
            # NaN placeholders instead of skipping the measurements completely
            # or registering zeros
            stats[key] = float('NaN')

    return stats


def get_percentiles(data, sigma):

    """Compute percentiles for data and return an array with the same length
    as the number of elements in ``sigma``.

    Args:
        data (array): 1-dimensional NumPy or CuPy arryay.
        sigma (tuple): Sigmas for which percentiles are computed.

    Returns:
        array: Array of percentiles.
    """

    def _get_percentiles(_data, _sigma):
        try:
            return np.percentile(_data, _sigma)
        except IndexError:  # Handle uninitialized model parameters
            return np.array((float('NaN'),) * 7)

    if isinstance(data, cupy.ndarray):
        # TODO(hvy): Make percentile computation faster for GPUs
        data = cupy.asnumpy(data)
        return cupy.asarray(_get_percentiles(data, sigma))

    return _get_percentiles(data, sigma)


def Monitor(base_name="main"):
    """Returns a trainer extension to monitor a model.

    This extension  calls the `monitor` method of a model
    each epoch.

    Note:
        Not used. Here for reference.
    """

    @extension.make_extension(
        trigger=(1, 'epoch'), priority=extension.PRIORITY_WRITER)
    def _monitor_model(trainer):
        trainer.updater.get_all_optimizers()[base_name].target.predictor.monitor()

    return _monitor_model

