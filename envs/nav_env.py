import numpy as np
import numpy.random as npr

class NavEnv(object):
  def __init__(self, seed, workspace, grid_world='easy', max_steps=10):
    super(NavEnv, self).__init__()

    self.seed = seed
    self.workspace = workspace
    self.max_steps = max_steps
    self.grid_world = grid_world

  def reset(self):
    if self.grid_world == 'easy':
      self.obs = self._loadEasyWorld()

    self.pos = self._generateValidCell()
    self.obs[self.pos[0], self.pos[1], 0] = 1

    self.goal = self._generateValidCell()
    self.obs[self.goal[0], self.goal[1], 1] = 1

    return self.obs

  def step(self, action):
    pass

  def _generateValidCell(self):
    is_cell_valid = False
    while not is_cell_valid:
      cell = npr.randint(self.grid_size, size=2)
      is_cell_valud = np.any(np.equal(self.occupied_cells, cell).all(1))

    self.occupied_cells.append(cell)
    return cell

  def _loadEasyWorld(self):
    self.grid_size = 5
    self.obs_size = self.grid_size * 10
    obs = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.float32)

    # Load walls around the world
    obs[:, 0,2] = 1
    obs[-1,0,2] = 1
    obs[0, :,2] = 1
    obs[0,-1,2] = 1

    # Set the occupied cells to the walls in the world
    self.occupied_cells = np.vstack(np.where(obs[:,:,2] > 0)).T

    return obs
