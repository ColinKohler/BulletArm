import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import ray
import argparse

from configs import *
from bulletarm_baselines.vtt.vtt.runner import Runner

task_configs = {
  'block_reaching' : BlockReachingConfig,
  'block_picking' : BlockPickingConfig,
  'block_pulling' : BlockPullingConfig,
  'block_pushing' : BlockPushingConfig,
  'block_pulling_corner' : BlockPullingCornerConfig,
  'peg_insertion' : PegInsertionConfig,
  'drawer_opening' : DrawerOpeningConfig,
  'drawer_closing' : DrawerClosingConfig,
  'clutter_picking' : ClutterPickingConfig,
  'mug_picking' : MugPickingConfig,
}

if __name__ == '__main__':
  parser=  argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('--num_gpus', type=int, default=1,
    help='Number of GPUs to use for training.')
  parser.add_argument('--results_path', type=str, default=None,
    help='Path to save results & logs to while training. Defaults to current timestamp.')
  parser.add_argument('--vision_size', type=int, default=64,
    help='The size of the RGB-D image used for vision.')
  parser.add_argument('--encoder', type=str, default='VTT',
    help='Type of latent encoder to use: VTT, POE, CONCAT')
  parser.add_argument('--checkpoint', type=str, default=None,
    help='Path to the checkpoint to load.')
  parser.add_argument('--buffer', type=str, default=None,
    help='Path to the replay buffer to load.')
  parser.add_argument('--log_file', type=str, default=None,
    help='Path to the log_file to load.')
  args = parser.parse_args()

  task_config = task_configs[args.task](args.vision_size, args.encoder, args.num_gpus, results_path=args.results_path)
  runner = Runner(task_config, checkpoint=args.checkpoint, replay_buffer=args.buffer, log_file=args.log_file)

  runner.train()
  ray.shutdown()
