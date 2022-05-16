import numpy as np
from bulletarm_baselines.fc_dqn.agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from bulletarm_baselines.fc_dqn.agents.agents_3d.margin_3d_fcn import Margin3DFCN

class Margin3DFCNSingleIn(DQN3DFCNSingleIn, Margin3DFCN):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), margin='l', margin_l=0.1, margin_weight=0.1,
                 softmax_beta=100):
        Margin3DFCN.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range, margin, margin_l, margin_weight, softmax_beta)
        DQN3DFCNSingleIn.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def update(self, batch):
        return Margin3DFCN.update(self, batch)

