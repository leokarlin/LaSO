==========
Experiment
==========


.. image:: https://img.shields.io/pypi/v/experiment.svg
        :target: https://pypi.python.org/pypi/experiment

.. image:: https://img.shields.io/travis/amitibo/experiment.svg
        :target: https://travis-ci.org/amitibo/experiment

.. image:: https://readthedocs.org/projects/experiment/badge/?version=latest
        :target: https://experiment.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Framework for running experiments.

The `experiment` package is meant for simplifying conducting experiments by hiding
most of the "boring" boiler plate code, e.g. experiment configuration and logging.
It is based on the Traitlets_ package.

.. note::
        The `experiment` package is still in beta state and the API might change.

* Free software: MIT license

.. * Documentation: https://pages.github.ibm.com/AMITAID/experiment/


TL;DR
-----

Copy the following example to a python file ``hello_experiment.py``::


    from experiment import Experiment
    import logging
    import time
    from traitlets import Int, Unicode


    class Main(Experiment):
        description = Unicode("My hellow world experiment.")
        epochs = Int(10, config=True, help="Number of epochs")

        def run(self):
            """Running the experiment"""

            logging.info("Starting experiment")

            loss = 100
            for i in range(self.epochs):
                logging.info("Running epoch [{}/[]]".format(i, self.epochs))
                time.sleep(.5)

            logging.info("Experiment finished")


    if __name__ == "__main__":
        main = Main()
        main.initialize()
        main.start()

Run the script from the command line like::

    $ python hello_experiment.py --epochs 15

The configuration, logs and results of the script will be stored in a unique folder under ``/tmp/results/...``.

To check the script documentation, run the following from the command line::

    $ python hello_experiment.py --help

See the documentation for more advanced usage.

Features
--------

* Clean and versatile configuration system based on the Traitlets_ package.
* Automatic logging setup.
* Configuration and logging are automatically saved in a unique results folder.
* Run parameters are stored in a configuraiton file to allow for replaying the same experiment.
* Support for multiple logging frameworks: mlflow_, visdom_, tensorboard_
* Automatic monitoring of GPU usage.

The examples_ folder contains multiple examples showcasing the package features.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Traitlets: https://traitlets.readthedocs.io/en/stable/index.html
.. _mlflow: https://mlflow.org/
.. _visdom: https://github.com/facebookresearch/visdom
.. _tensorboard: https://www.tensorflow.org/guide/summaries_and_tensorboard
.. _examples: https://github.ibm.com/AMITAID/experiment/tree/master/examples
