from __future__ import division
import numpy as np
import os
from pkg_resources import resource_filename
import warnings


RESOURCES_BASE = os.path.abspath(resource_filename(__name__, '../resources'))

try:
    FACEID_BASE = os.environ['FACEID_BASE']
except Exception:
    FACEID_BASE = 'C:'
    warnings.warn('Failed to find find FACEID_BASE environment variable')

RESULTS_HOME = os.path.join(FACEID_BASE, 'results')
TENSORBOARD_HOME = os.path.join(FACEID_BASE, 'tensorboard')
