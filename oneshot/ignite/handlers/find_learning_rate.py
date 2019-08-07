from ignite.engine import Events
from ignite.metrics import Metric
import math


class LRFinder(Metric):
    """"""

    def __init__(self, optimizer, init_lr=1e-8, final_lr=10., beta=0.98, output_transform=lambda x: x):

        self.init_lr = init_lr
        self.final_lr = final_lr
        self.beta = beta

        self.optimizer = optimizer

        super(LRFinder, self).__init__(output_transform=output_transform)

    def attach(self, engine, name):
        """ Register callbacks to control the search for learning rate.

        Args:
            engine (ignite.engine.Engine):
                Engine that this handler will be attached to

        Returns:
            self (Timer)

        """

        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

        self.engine = engine

        return self

    def _update_optimizer_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.lr

    def reset(self):
        """
        Resets the metric to to it's initial state.

        This is called at the start of each epoch.
        """
        self.lr = self.init_lr
        self._update_optimizer_lr()

        self.avg_loss = 0

    def update(self, output):
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function
        """
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * output

        self.lr *= self.mult
        self._update_optimizer_lr()

    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest

        Raises:
            NotComputableError: raised when the metric cannot be computed
        """
        #
        # Compute the smoothed loss
        #
        smoothed_loss = self.avg_loss / (1 - self.beta ** (self.engine.state.iteration + 1))

        #
        # Stop if the loss is exploding
        #
        if self.engine.state.iteration > 1 and smoothed_loss > 4 * self.best_loss:
            self.engine.terminate()

        #
        # Record the best loss
        #
        if self.engine.state.iteration == 1 or smoothed_loss < self.best_loss:
            self.best_loss = smoothed_loss

        #
        # Store the values
        #
        self.log_lr = math.log10(self.lr)

        return smoothed_loss

    def started(self, engine):

        self.mult = (self.final_lr / self.init_lr) ** (1 / len(engine.state.dataloader))
