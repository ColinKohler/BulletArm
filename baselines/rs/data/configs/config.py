import sys
sys.path.append('..')

import os
import datetime
import math
import numpy as np

from data import constants

class Config(object):
  def __init__(self, num_gpus=1):
    self.num_gpus = num_gpus
    self.num_sampler_workers = constants.GPU_CONFIG[self.num_gpus]['num_sampler_workers']
    self.gen_data_on_gpu = False
    self.per_beta_anneal_steps = None

  def getPerBeta(self, step):
    anneal_steps = self.per_beta_anneal_steps if self.per_beta_anneal_steps else self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_per_beta - self.end_per_beta) * r + self.end_per_beta
