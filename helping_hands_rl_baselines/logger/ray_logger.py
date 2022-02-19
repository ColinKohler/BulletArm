import ray

from helping_hands_rl_baselines.logger.logger import Logger

@ray.remote
class RayLogger(Logger):
  def __init__(self, results_path, hyperparameters):
    super().__init__(results_path, hyperparameters)
