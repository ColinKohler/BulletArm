import argparse

from bulletarm_baselines.logger.plotter import Plotter

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  args = parser.parse_args()

  log_filepaths = [
    '/home/colin/hdd/workspace/midichlorians/data/block_pulling/baseline/log_data.pkl',
    '/home/colin/hdd/workspace/ysalamir/data/block_pulling/baseline/log_data.pkl',
  ]
  log_names = ['force', 'vanilla']

  plotter = Plotter(log_filepaths, log_names)
  plotter.plotLearningCurves('128x128 Block Picking', 'test.pdf', window=500, max_eps=None)
