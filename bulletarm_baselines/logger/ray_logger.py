import ray

from bulletarm_baselines.logger.logger import Logger

@ray.remote
class RayLogger(Logger):
  def __init__(self, results_path, num_eval_eps=100, hyperparameters=None):
    super().__init__(results_path, num_eval_eps=num_eval_eps, hyperparameters=hyperparameters)
