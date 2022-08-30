import argparse

from bulletarm_baselines.logger.plotter import Plotter

def plot(lc_smoothing=100, eval_smoothing=5, num_eval=None):
  log_filepaths = [
    '/home/colin/hdd/workspace/midichlorians/data/peg_insertion/gate_better_physics_64/log_data.pkl',
    '/home/colin/hdd/workspace/ysalamir/data/peg_insertion/render_center_new_physics_64/log_data.pkl',
  ]
  log_names = ['vision+force', 'vision']
  titles = '64x64 Peg Insertion Depth Rendering'

  base_dir = 'scripts/outputs/'

  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves(titles, base_dir + 'train.pdf', window=lc_smoothing)
  plotter.plotEvalRewards(titles, base_dir + 'eval_rewards.pdf', window=eval_smoothing, num_eval_intervals=num_eval, eval_interval=500)
  plotter.plotEvalReturns(titles, base_dir +'eval_returns.pdf', window=eval_smoothing, num_eval_intervals=num_eval, eval_interval=500)
  plotter.plotEvalLens(titles, base_dir + 'eval_lens.pdf', window=eval_smoothing)
  plotter.plotEvalValues(titles, base_dir +'eval_values.pdf', window=eval_smoothing)

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
