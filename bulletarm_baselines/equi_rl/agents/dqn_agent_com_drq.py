import numpy as np
import torch
import torch.nn.functional as F
from bulletarm_baselines.equi_rl.agents.dqn_agent_com import DQNAgentCom
from bulletarm_baselines.equi_rl.utils.torch_utils import augmentTransition
from bulletarm_baselines.equi_rl.utils.parameters import obs_type

class DQNAgentComDrQ(DQNAgentCom):
    """
    Class for DrQ DQN agent
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.K = 2
        self.M = 2

    def _loadBatchToDevice(self, batch):
        """
        Load batch into pytorch tensor. Perform K augmentations on next observation and M augmentations on current
        observation and action
        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        """
        K_next_obs = []
        M_obs = []
        M_action = []
        for _ in range(self.K):
            for d in batch:
                K_aug_d = augmentTransition(d, 'dqn_c4')
                K_next_obs.append(torch.tensor(K_aug_d.next_obs))
        for _ in range(self.M):
            for d in batch:
                M_aug_d = augmentTransition(d, 'dqn_c4')
                M_obs.append(torch.tensor(M_aug_d.obs))
                M_action.append(torch.tensor(M_aug_d.action))

        K_next_obs_tensor = torch.stack(K_next_obs).to(self.device)
        M_obs_tensor = torch.stack(M_obs).to(self.device)
        M_action_tensor = torch.stack(M_action).to(self.device)

        if obs_type is 'pixel':
            K_next_obs_tensor = K_next_obs_tensor/255*0.4
            M_obs_tensor = M_obs_tensor/255*0.4

        self.loss_calc_dict['K_next_obs'] = K_next_obs_tensor
        self.loss_calc_dict['M_obs'] = M_obs_tensor
        self.loss_calc_dict['M_action'] = M_action_tensor

        return super()._loadBatchToDevice(batch)

    def calcTDLoss(self):
        """
        Calculate the TD loss
        :return: td loss, td error
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        K_next_obs = self.loss_calc_dict['K_next_obs']
        M_obs = self.loss_calc_dict['M_obs']
        M_action = self.loss_calc_dict['M_action']
        p_id = M_action[:, 0]
        dxy_id = M_action[:, 1]
        dz_id = M_action[:, 2]
        dtheta_id = M_action[:, 3]

        with torch.no_grad():
            q_all_prime = self.forwardNetwork(next_states.repeat(self.K), K_next_obs, target_net=True)
            q_prime = q_all_prime.reshape(batch_size*self.K, -1).max(1)[0]
            q_target = rewards.repeat(self.K) + self.gamma * q_prime * non_final_masks.repeat(self.K)
            q_target = q_target.reshape(self.K, batch_size).mean(dim=0)

        q = self.forwardNetwork(states.repeat(self.M), M_obs)
        q_pred = q[torch.arange(batch_size*self.M), dxy_id, dz_id, dtheta_id, p_id]
        self.loss_calc_dict['q_output'] = q
        self.loss_calc_dict['q_pred'] = q_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target.repeat(self.M))
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target.repeat(self.M)).reshape(self.M, batch_size).mean(dim=0)
        return td_loss, td_error
