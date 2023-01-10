import pybullet as pb
import numpy as np

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.equipment.cabinet import Cabinet

class CloseLoopCabinetOpeningEnv(CloseLoopEnv):
  '''Close loop cabinet opening task.

  The robot needs to pull the handle of the cabinet to open it.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    super().__init__(config)
    self.cabinet = Cabinet()
    self.cabinet_rot = 0

  def initialize(self):
    super().initialize()
    self.cabinet.initialize((self.workspace[0].mean() + 0.05, self.workspace[1].mean(), 0), pb.getQuaternionFromEuler((0, 0, 0)))

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    pos = np.array([self.workspace[0].mean() + 0.05, self.workspace[1].mean(), 0])
    self.cabinet_rot = 3*np.pi/2 # np.random.random()*2*np.pi if self.random_orientation else np.random.choice([np.pi/2, 3*np.pi/2])
    #m = np.array(transformations.euler_matrix(0, 0, self.cabinet_rot))[:3, :3]
    #dx = np.random.random() * (0.1 - 0.1) + 0.1
    #dy = np.random.random() * (0.1 - -0.1) + -0.1
    #pos = pos + m[:, 0] * dx
    #pos = pos + m[:, 1] * dy
    self.cabinet.reset(pos, transformations.quaternion_from_euler(0, 0, self.cabinet_rot))

    return self._getObservation()

  def _checkTermination(self):
    return self.cabinet.isOpen()
