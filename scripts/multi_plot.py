import argparse

from bulletarm_baselines.logger.multi_plotter import MultiPlotter

def plot(lc_eps=None, lc_smoothing=100, eval_smoothing=5, num_eval=None):
  log_filepaths = [
    [
      '/home/colin/hdd/workspace/discovery_runs/drawer_opening/force_1/log_data.pkl',
      '/home/colin/hdd/workspace/discovery_runs/drawer_opening/force_2/log_data.pkl',
      '/home/colin/hdd/workspace/discovery_runs/drawer_opening/force_3/log_data.pkl',
      #'/home/colin/hdd/workspace/discovery_runs/block_picking/0_5_occlusion_snp_force_1/log_data.pkl',
      #'/home/colin/hdd/workspace/discovery_runs/block_picking/0_5_occlusion_snp_force_2/log_data.pkl',
      #'/home/colin/hdd/workspace/discovery_runs/block_picking/0_5_occlusion_snp_force_3/log_data.pkl',
      #'/home/colin/hdd/workspace/midichlorians/data/block_picking/low_res_gitter_sensor_test/log_data.pkl',
      #'/home/colin/hdd/workspace/midichlorians/data/block_picking/low_res_gitter_sensor_test_2/log_data.pkl',
      #'/home/colin/hdd/workspace/midichlorians/data/block_picking/low_res_gitter_sensor_test_3/log_data.pkl',
    ],
    [
      '/home/colin/hdd/workspace/discovery_runs/drawer_opening/vision_1/log_data.pkl',
      '/home/colin/hdd/workspace/discovery_runs/drawer_opening/vision_2/log_data.pkl',
      '/home/colin/hdd/workspace/discovery_runs/drawer_opening/vision_3/log_data.pkl',
      #'/home/colin/hdd/workspace/discovery_runs/block_picking/0_5_occlusion_snp_vision_1/log_data.pkl',
      #'/home/colin/hdd/workspace/discovery_runs/block_picking/0_5_occlusion_snp_vision_2/log_data.pkl',
      #'/home/colin/hdd/workspace/discovery_runs/block_picking/0_5_occlusion_snp_vision_3/log_data.pkl',
      #'/home/colin/hdd/workspace/ysalamir/data/block_picking/low_res_gitter_sensor_test/log_data.pkl',
      #'/home/colin/hdd/workspace/ysalamir/data/block_picking/low_res_gitter_sensor_test_2/log_data.pkl',
    ]
  ]
  log_names = [
    'vision+force',
    'vision'
  ]
  title = '128x128 Drawer Opening'
  base_dir = 'scripts/outputs/'

  plotter = MultiPlotter(log_filepaths, log_names)
  #plotter.plotLearningCurves(title, base_dir + 'train.pdf', max_eps=lc_eps, window=lc_smoothing)
  plotter.plotEvalRewards(title, base_dir + 'eval_rewards.pdf', num_eval_intervals=num_eval, window=eval_smoothing, eval_interval=500)
  plotter.plotEvalLens(title, base_dir + 'eval_lens.pdf', num_eval_intervals=num_eval, window=eval_smoothing, eval_interval=500)

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  parser.add_argument('--lc_eps', type=int, default=None,
    help='Number of episodes to plot on the learning curve')
  parser.add_argument('--lc_smoothing', type=int, default=100,
    help='Window size for learning curve averaging.')
  parser.add_argument('--num_eval', type=int, default=None,
    help='Number of intervals to plot on the evaluation curve')
  parser.add_argument('--eval_smoothing', type=int, default=1,
    help='Window size for evaluation curve averaging.')
  args = parser.parse_args()

  plot(args.lc_eps, args.lc_smoothing, args.eval_smoothing, args.num_eval)