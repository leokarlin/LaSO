import numpy as np
import torch

from ignite.engine import Engine


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_mixup_trainer(model, optimizer, loss_fn, alpha=1.0, device=None):
    """
    Factory function for creating a trainer for mixup augmented models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(engine, batch):

        from ignite.engine import _prepare_batch

        model.train()
        optimizer.zero_grad()

        inputs, targets = _prepare_batch(batch, device=device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha, use_cuda=(device=="cuda"))
        outputs = model(inputs)

        loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_mixup_trainer_x2(model, optimizer, loss_fn, alpha=1.0, device=None):
    """
    Factory function for creating a trainer for mixup augmented models.
    It expects that the model outputs two outputs (like the Inception3 auxlogits).

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(engine, batch):

        from ignite.engine import _prepare_batch

        model.train()
        optimizer.zero_grad()

        inputs, targets = _prepare_batch(batch, device=device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha, use_cuda=(device=="cuda"))
        outputs1, outputs2 = model(inputs)

        loss1 = mixup_criterion(loss_fn, outputs1, targets_a, targets_b, lam)
        loss2 = mixup_criterion(loss_fn, outputs2, targets_a, targets_b, lam)

        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


