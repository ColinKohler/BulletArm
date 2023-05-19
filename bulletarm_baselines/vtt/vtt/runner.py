import os
import shutil
import time
import copy
import ray
import torch
import numpy.random as npr
import pickle
import time
from bulletarm_baselines.vtt.vtt.trainer import Trainer
from bulletarm_baselines.vtt.vtt.replay_buffer import ReplayBuffer
from bulletarm_baselines.vtt.vtt.data_generator import DataGenerator, EvalDataGenerator
from bulletarm_baselines.vtt.vtt.shared_storage import SharedStorage

from bulletarm_baselines.logger.logger import RayLogger

class Runner(object):
  '''
  Runner class. Used to train the model and log the results to Tensorboard.

  Args:
    config (dict): Task configuration
    checkpoint (str): Path to checkpoint to load after initialization. Defaults to None.
    replay_buffer (dict): Path to replay buffer to load after initialization. Defaults to None.
  '''
  def __init__(self, config, checkpoint=None, replay_buffer=None):
    self.config = config

    # Set random seeds
    if self.config.seed:
      npr.seed(self.config.seed)
      torch.manual_seed(self.config.seed)
    ray.init(num_gpus=self.config.num_gpus, ignore_reinit_error=True)

    # Create log dir
    if os.path.exists(self.config.results_path):
      shutil.rmtree(self.config.results_path)
    os.makedirs(self.config.results_path)

    # Initialize checkpoint and replay buffer
    self.checkpoint = {
      'best_weights' : None,
      'weights' : None,
      'optimizer_state' : None,
      'training_step' : 0,
      'latent_training_step' : 0,
      'num_eps' : 0,
      'num_steps' : 0,
      'best_model_reward' : 0,
      'run_eval_interval' : False,
      'generating_eval_eps' : False,
      'pause_training' : False,
      'terminate' : False
    }
    self.replay_buffer = dict()

    # Load checkpoint/replay buffer
    if checkpoint:
      checkpoint = os.path.join(self.config.root_path,
                                checkpoint,
                                'model.checkpoint')
    if replay_buffer:
      replay_buffer = os.path.join(self.config.root_path,
                                   replay_buffer,
                                   'replay_buffer.pkl')
    self.load(checkpoint_path=checkpoint,
              replay_buffer_path=replay_buffer)
    
    # Workers
    self.logger_worker = None
    self.data_gen_workers = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None
    self.training_worker = None
    self.eval_worker = None

  def train(self):
    '''
    Initialize the various workers, start the trainers, and run the logging loop.
    '''
    self.logger_worker = RayLogger.options(num_cpus=0, num_gpus=0).remote(
      self.config.results_path,
      self.config.__dict__,
      checkpoint_interval=self.config.checkpoint_interval,
      num_eval_eps=self.config.num_eval_episodes
    )
    self.training_worker = Trainer.options(num_cpus=0, num_gpus=1).remote(self.checkpoint, self.config)

    self.replay_buffer_worker = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(
      self.checkpoint,
      self.replay_buffer,
      self.config
    )
    self.eval_worker = EvalDataGenerator.options(num_cpus=0, num_gpus=0).remote(
      self.config,
      self.config.seed+self.config.num_data_gen_envs if self.config.seed else None
    )

    self.shared_storage_worker = SharedStorage.remote(self.checkpoint, self.config)
    self.shared_storage_worker.setInfo.remote('terminate', False)

    # Blocking call to generate expert data
    if self.config.num_expert_episodes > 0:
      self.training_worker.generateExpertData.remote(self.replay_buffer_worker, self.shared_storage_worker, self.logger_worker)
    else:
      self.training_worker.generateData.remote(self.config.num_data_gen_envs, self.replay_buffer_worker, self.shared_storage_worker, self.logger_worker)

    # Start training
    self.training_worker.continuousUpdateWeights.remote(self.replay_buffer_worker, self.shared_storage_worker, self.logger_worker)

    self.loggingLoop()

  def loggingLoop(self):
    '''
    Initialize the testing model and log the training data
    '''
    self.save(logging=True)

    # Log training loop
    keys = [
      'training_step',
      'run_eval_interval',
      'generating_eval_eps'
    ]

    start_time = time.time()
    timeout_soon = 7.9 * 60 * 60
    info = ray.get(self.shared_storage_worker.getInfo.remote(keys))
    try:
      while info['training_step'] < self.config.training_steps or info['generating_eval_eps'] or info['run_eval_interval']:
        # if time.time() - start_time > timeout_soon:
        #   self.logger_worker.exportData.remote()
        info = ray.get(self.shared_storage_worker.getInfo.remote(keys))

        # Eval
        if info['run_eval_interval']:
          if info['generating_eval_eps']:
            self.shared_storage_worker.setInfo.remote('pause_training', True)
          while(ray.get(self.shared_storage_worker.getInfo.remote('generating_eval_eps'))):
            time.sleep(0.1)
          self.shared_storage_worker.setInfo.remote('pause_training', False)
          self.eval_worker.generateEpisodes.remote(self.config.num_eval_episodes, self.shared_storage_worker, self.replay_buffer_worker, self.logger_worker)

        # Logging
        self.logger_worker.writeLog.remote()

        time.sleep(0.5)
    except KeyboardInterrupt:
      pass

    if self.config.save_model:
      self.shared_storage_worker.setInfo.remote(copy.copy(self.replay_buffer))
      self.logger_worker.exportData.remote()
    self.terminateWorkers()

  def save(self, logging=False):
    '''
    Save the model checkpoint and replay buffer.

    Args:
      logging (bool): Print logging string when saving. Defaults to False.
    '''
    if logging:
      print('Checkpointing model at: {}'.format(self.config.results_path))
    self.shared_storage_worker.saveReplayBuffer.remote(copy.copy(self.replay_buffer))
    self.shared_storage_worker.saveCheckpoint.remote()

  def load(self, checkpoint_path=None, replay_buffer_path=None):
    '''
    Load the model checkpoint and replay buffer.

    Args:
      checkpoint_path (str): Path to model checkpoint to load. Defaults to None.
      replay_buffer_path (str): Path to replay buffer to load. Defaults to None.
    '''
    if checkpoint_path:
      if os.path.exists(checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path)
        print('Loading checkpoint from {}'.format(checkpoint_path))
      else:
        print('Checkpoint not found at {}'.format(checkpoint_path))

    if replay_buffer_path:
      if os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, 'rb') as f:
          data = pickle.load(f)

        self.replay_buffer = data['buffer']
        self.checkpoint['num_eps'] = data['num_eps']
        self.checkpoint['num_steps'] = data['num_steps']

        print('Loaded replay buffer at {}'.format(replay_buffer_path))
      else:
        print('Replay buffer not found at {}'.format(replay_buffer_path))

  def terminateWorkers(self):
    '''
    Terminate the various workers.
    '''
    if self.shared_storage_worker:
      self.shared_storage_worker.setInfo.remote('terminate', True)
      self.checkpoint = ray.get(self.shared_storage_worker.getCheckpoint.remote())
    if self.replay_buffer_worker:
      self.replay_buffer = ray.get(self.replay_buffer_worker.getBuffer.remote())

    self.logger_worker = None
    self.data_gen_workers = None
    self.test_worker = None
    self.training_worker = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None
