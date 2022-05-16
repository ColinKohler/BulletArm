import numpy as np
import torch
import time
from functools import wraps
from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L
from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l_deictic import DQN6DASR5LDeictic

class DQN6DASR5LDeictic35(DQN6DASR5LDeictic):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), num_ry=8, ry_range=(0, 7 * np.pi / 8), num_rx=8,
                 rx_range=(0, 7 * np.pi / 8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range,
                         num_ry, ry_range, num_rx, rx_range, num_zs, z_range)

    def getQ2Input(self, obs, center_pixel):
        return DQN6DASR5L.getQ2Input(self, obs, center_pixel)

    def forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net=False, to_cpu=False):
        return DQN6DASR5L.forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net, to_cpu)