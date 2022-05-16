from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from bulletarm_baselines.fc_dqn.agents.base_agent import BaseAgent
from bulletarm_baselines.fc_dqn.utils import torch_utils


class DQN2DFCN(BaseAgent):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)

    def initNetwork(self, fcn):
        self.fcn = fcn
        self.target_fcn = deepcopy(fcn)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.fcn)
        self.target_networks.append(self.target_fcn)
        self.optimizers.append(self.fcn_optimizer)
        self.updateTarget()

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.):
        with torch.no_grad():
            q_value_maps, _ = self.forwardFCN(states, in_hand, obs, to_cpu=True)
        pixels = torch_utils.argmax2d(q_value_maps).long()

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        for i, m in enumerate(rand_mask):
            if m:
                pixel_candidates = torch.nonzero(obs[i, 0] > 0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        actions = torch.cat((x, y), dim=1)
        action_idx = pixels
        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        actions = torch.cat((x, y), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y), dim=1)
        return action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]

        with torch.no_grad():
            q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs[1], next_obs[0], target_net=True)
            x_star = torch_utils.argmax2d(q1_map_prime).long()
            q_prime = q1_map_prime[torch.arange(0, batch_size), x_star[:, 0], x_star[:, 1]]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        self.loss_calc_dict['q1_output'] = q1_output

        q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
        td_loss = q1_td_loss

        with torch.no_grad():
            td_error = torch.abs(q1_pred - q_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
