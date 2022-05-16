import numpy as np
import torch
import torch.nn.functional as F
from bulletarm_baselines.equi_rl.agents.dqn_base import DQNBase
from bulletarm_baselines.equi_rl.utils import torch_utils

class DQNAgentCom(DQNBase):
    """
    Class for DQN (composed) agent
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.n_xy = 9
        self.n_z = 3
        self.n_theta = n_theta
        self.n_p = n_p

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        """
        Forward pass the Q-network
        :param state: gripper state
        :param obs: observation
        :param target_net: whether to use the target network
        :param to_cpu: move output to cpu
        :return: the output of the Q-network
        """
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q = net(stacked.to(self.device))
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_xy, self.n_z, self.n_theta, self.n_p)
        return q

    def getEGreedyActions(self, state, obs, eps):
        """
        Get e-greedy actions
        :param state: gripper holding state
        :param obs: observation
        :param eps: epsilon
        :return: action ids, actions
        """
        with torch.no_grad():
            q = self.forwardNetwork(state, obs, to_cpu=True)
            argmax = torch_utils.argmax4d(q)
            dxy_id = argmax[:, 0]
            dz_id = argmax[:, 1]
            dtheta_id = argmax[:, 2]
            p_id = argmax[:, 3]

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_p = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_p)
        p_id[rand_mask] = rand_p.long()
        rand_dxy = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_xy)
        dxy_id[rand_mask] = rand_dxy.long()
        rand_dz = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_z)
        dz_id[rand_mask] = rand_dz.long()
        rand_dtheta = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_theta)
        dtheta_id[rand_mask] = rand_dtheta.long()
        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def calcTDLoss(self):
        """
        Calculate the TD loss
        :return: td loss, td error
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_all_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_all_prime.reshape(batch_size, -1).max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q = self.forwardNetwork(states, obs)
        q_pred = q[torch.arange(batch_size), dxy_id, dz_id, dtheta_id, p_id]
        self.loss_calc_dict['q_output'] = q
        self.loss_calc_dict['q_pred'] = q_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        return td_loss, td_error
