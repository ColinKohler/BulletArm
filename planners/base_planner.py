

class BasePlanner(object):
  def __init__(self, env):
    self.env = env

  def getNextAction(self):
    raise NotImplemented('Planners must implement this function')
