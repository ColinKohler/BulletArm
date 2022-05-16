import numpy as np
import torch
import torch.nn.functional as F
from bulletarm_baselines.fc_dqn.agents.agents_3d.dqn_3d_fcn import DQN3DFCN

class DQN3DFCNSingleIn(DQN3DFCN):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        DQN3DFCN.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        q1 = self.fcn if not target_net else self.target_fcn
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps, obs_encoding = q1(obs, in_hand)
        q_value_maps = q_value_maps.reshape((states.shape[0], self.num_primitives, self.num_rz, q_value_maps.size(-2), q_value_maps.size(-1)))
        if padding_width > 0:
            q_value_maps = q_value_maps[torch.arange(0, states.size(0)), states.long(), :, padding_width: -padding_width, padding_width: -padding_width]
        else:
            q_value_maps = q_value_maps[torch.arange(0, states.size(0)), states.long()]
        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

