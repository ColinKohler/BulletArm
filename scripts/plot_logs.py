import argparse

from bulletarm_baselines.logger.plotter import Plotter

def plot(lc_smoothing=100, eval_smoothing=5, num_eval=None):
  log_filepaths = [
    '/home/colin/hdd/workspace/discovery_runs/block_picking/0_1_3_occlusions_force_1/log_data.pkl',
    '/home/colin/hdd/workspace/discovery_runs/block_picking/0_1_3_occlusions_force_2/log_data.pkl',
    '/home/colin/hdd/workspace/discovery_runs/block_picking/0_1_3_occlusions_force_3/log_data.pkl',
    '/home/colin/hdd/workspace/discovery_runs/block_picking/0_1_3_occlusions_vision_1/log_data.pkl',
    '/home/colin/hdd/workspace/discovery_runs/block_picking/0_1_3_occlusions_vision_2/log_data.pkl',
    '/home/colin/hdd/workspace/discovery_runs/block_picking/0_1_3_occlusions_vision_3/log_data.pkl',
  ]
  log_names = [
    'force1',
    'force2',
    'force3',
    'vision1',
    'vision2',
    'vision3'
  ]
  title = '128x128 Block Picking'
  base_dir = 'scripts/outputs/'

  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves(title, base_dir + 'train.pdf', window=lc_smoothing)
  plotter.plotEvalRewards(title, base_dir + 'eval_rewards.pdf', window=eval_smoothing, num_eval_intervals=num_eval, eval_interval=500)
  plotter.plotEvalReturns(title, base_dir +'eval_returns.pdf', window=eval_smoothing, num_eval_intervals=num_eval, eval_interval=500)
  plotter.plotEvalLens(title, base_dir + 'eval_lens.pdf', window=eval_smoothing)
  plotter.plotEvalValues(title, base_dir +'eval_values.pdf', window=eval_smoothing)

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  parser.add_argument('--lc_smoothing', type=int, default=100,
    help='The amount of smoothing to apply to the learning curve.')
  parser.add_argument('--eval_smoothing', type=int, default=5,
    help='The amount of smoothing to applyt to the evaluation curves.')
  parser.add_argument('--num_eval', type=int, default=None,
    help='The number of eval intervals to plot. Defaults to all.')
  args = parser.parse_args()

  plot(args.lc_smoothing, args.eval_smoothing, args.num_eval)
