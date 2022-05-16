import numpy as np

from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l_deictic_35 import DQN6DASR5LDeictic35
from bulletarm_baselines.fc_dqn.agents.agents_6d.margin_6d_asr_5l import Margin6DASR5L

class Margin6DASR5LDeictic35(DQN6DASR5LDeictic35, Margin6DASR5L):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), num_ry=8, ry_range=(0, 7*np.pi/8), num_rx=8,
                 rx_range=(0, 7*np.pi/8), num_zs=16, z_range=(0.02, 0.12),
                 margin='l', margin_l=0.1, margin_weight=0.1, softmax_beta=100):
        Margin6DASR5L.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                               num_rz, rz_range, num_ry, ry_range, num_rx, rx_range, num_zs, z_range, margin, margin_l,
                               margin_weight, softmax_beta)
        DQN6DASR5LDeictic35.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                     num_rz, rz_range, num_ry, ry_range, num_rx, rx_range, num_zs, z_range)

    def update(self, batch):
        return Margin6DASR5L.update(self, batch)
