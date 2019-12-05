import numpy as np
import numpy.random as npr

class BasePlanner(object):
  def __init__(self, env, config):
    self.env = env
    self.rand_pick_prob = config['rand_pick_prob'] if 'rand_pick_prob' in config else 0.0
    self.rand_place_prob = config['rand_place_prob'] if 'rand_place_prob' in config else 0.0
    self.pos_noise = config['pos_noise'] if 'pos_noise' in config else None
    self.rot_noise = config['rot_noise'] if 'rot_noise' in config else None

  def getNextAction(self):
    raise NotImplemented('Planners must implement this function')

  def addNoiseToPos(self, x, y):
    # TODO: Would we ever want to include noise on the z-axis here?
    if self.pos_noise:
      x = np.clip(x + npr.uniform(-self.pos_noise, self.pos_noise), self.env.workspace[0,0], self.env.workspace[0,1])
      y = np.clip(y + npr.uniform(-self.pos_noise, self.pos_noise), self.env.workspace[1,0], self.env.workspace[1,1])
    return x, y

  def addNoiseToRot(self, rot):
    if self.rot_noise:
      rot = np.clip(rot + npr.uniform(-self.rot_noise, self.rot_noise), 0., np.pi)
    return rot
