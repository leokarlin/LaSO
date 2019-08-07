from functools import partial
from statistics import mean
from typing import Union, Tuple, List
import torch

from ignite.metrics import EpochMetric
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class MultiLabelSoftMarginAccuracy(Metric):
    """
    Calculates the multi-label accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories)
    - `y` must be in the following shape (batch_size, num_categories)
    - sigmoid is applied before comparing to 0.5.
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()
        correct = torch.eq(y_pred, y).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MultiLabelSoftMarginAccuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class MultiLabelSoftMarginIOUaccuracy(Metric):
    """
    Calculates the multi-label accuracy. The accuracy is calculated only
    IOU of the predicted and ground truth.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories)
    - `y` must be in the following shape (batch_size, num_categories)
    - sigmoid is applied before comparing to 0.5.
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()

        support = (y + y_pred) > 0.5
        correct = torch.eq(y_pred, y)[support]

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MultiLabelSoftMarginIOUaccuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class ReductionMetric(Metric):
    """
    Apply reduction to a list of metrics.

    Args:
        metrics (list of metrics): The metrics to apply reduction to.
        reduction (callable): The reduction operation to apply to the metrics.
        output_transform (callable): a callable that is used to transform the
            model's output into the form expected by the metric. This can be
            useful if, for example, you have a multi-output model and you want to
            compute the metric with respect to one of the outputs.
    """
    def __init__(self, metrics: Union[Tuple, List], reduction=mean, output_transform=lambda x: x):

        self.metrics = metrics
        self.reduction = reduction
        super(ReductionMetric, self).__init__(output_transform=output_transform)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, output):
        for metric in self.metrics:
            o = metric._output_transform(output)
            metric.update(o)

    def compute(self):
        return self.reduction([m.compute() for m in self.metrics])


class EWMeanSquaredError(Metric):
    """
    Calculates the Element-Wise mean squared error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_elements = 0

    def update(self, output):
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        self._sum_of_squared_errors += torch.sum(squared_errors).item()
        self._num_elements += y.numel()

    def compute(self):
        if self._num_elements == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed')
        return self._sum_of_squared_errors / self._num_elements


class CallbackMetric(object):
    def __init__(self, name):
        pass


def average_precision_compute_fn(y_preds, y_targets, mask, activation=None):
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_pred = y_preds.numpy()

    if mask is not None:
        y_true = y_true[:, mask]
        y_pred = y_pred[:, mask]

    return average_precision_score(y_true, y_pred)


class mAP(EpochMetric):
    """Computes Average Precision accumulating predictions and the ground-truth during an epoch
    and applying `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ .

    Args:
        activation (callable, optional): optional function to apply on prediction tensors,
            e.g. `activation=torch.sigmoid` to transform logits.
        mask (bool array, optional): Ignore some of the classes (useful for filtering unseen classes).
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """
    def __init__(self, activation=None, mask=None, output_transform=lambda x: x):
        super(mAP, self).__init__(partial(average_precision_compute_fn, mask=mask, activation=activation),
                                               output_transform=output_transform)
