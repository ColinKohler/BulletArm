from bulletarm_baselines.equi_rl.agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from bulletarm_baselines.equi_rl.utils.torch_utils import augmentTransition

class SACDrQ(SAC):
    """
    DrQ SAC agent class
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi / 16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False, obs_type='pixel'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning, obs_type)
        self.K = 2
        self.M = 2
        self.aug_type = 'cn'

    def _loadBatchToDevice(self, batch):
        states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = super()._loadBatchToDevice(batch)
        K_next_obs = []
        M_obs = []
        M_action = []
        for _ in range(self.K):
            for d in batch:
                K_aug_d = augmentTransition(d, self.aug_type)
                K_next_obs.append(torch.tensor(K_aug_d.next_obs))
        for _ in range(self.M):
            for d in batch:
                M_aug_d = augmentTransition(d, self.aug_type)
                M_obs.append(torch.tensor(M_aug_d.obs))
                M_action.append(torch.tensor(M_aug_d.action))

        K_next_obs_tensor = torch.stack(K_next_obs).to(self.device)
        if self.obs_type is 'pixel':
            K_next_obs_tensor = torch.cat([K_next_obs_tensor, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(self.K, 1, K_next_obs_tensor.shape[2], K_next_obs_tensor.shape[3])], dim=1)
        M_obs_tensor = torch.stack(M_obs).to(self.device)
        if self.obs_type is 'pixel':
            M_obs_tensor = torch.cat([M_obs_tensor, states.reshape(states.size(0), 1, 1, 1).repeat(self.M, 1, M_obs_tensor.shape[2], M_obs_tensor.shape[3])], dim=1)
        M_action_tensor = torch.stack(M_action).to(self.device)

        if self.obs_type is 'pixel':
            K_next_obs_tensor = K_next_obs_tensor/255*0.4
            M_obs_tensor = M_obs_tensor/255*0.4

        self.loss_calc_dict['K_next_obs'] = K_next_obs_tensor
        self.loss_calc_dict['M_obs'] = M_obs_tensor
        self.loss_calc_dict['M_action'] = M_action_tensor

        return super()._loadBatchToDevice(batch)

    def calcActorLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_expert = self._loadLossCalcDict()
        M_obs = self.loss_calc_dict['M_obs']
        M_action = self.loss_calc_dict['M_action']
        pi, log_pi, mean = self.actor.sample(M_obs)
        qf1_pi, qf2_pi = self.critic(M_obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        is_experts = is_expert.repeat(self.M)

        self.loss_calc_dict['pi'] = pi
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi
        self.loss_calc_dict['action_idx'] = M_action
        self.loss_calc_dict['is_experts'] = is_experts

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        return policy_loss

    def calcCriticLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            K_next_obs = self.loss_calc_dict['K_next_obs']
            next_state_action, next_state_log_pi, _ = self.actor.sample(K_next_obs)
            next_state_log_pi = next_state_log_pi.reshape(self.K*batch_size)
            qf1_next_target, qf2_next_target = self.critic_target(K_next_obs, next_state_action)
            qf1_next_target = qf1_next_target.reshape(self.K*batch_size)
            qf2_next_target = qf2_next_target.reshape(self.K*batch_size)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards.repeat(self.K) + non_final_masks.repeat(self.K) * self.gamma * min_qf_next_target
            next_q_value = next_q_value.reshape(self.K, batch_size).mean(dim=0)

        M_obs = self.loss_calc_dict['M_obs']
        M_action = self.loss_calc_dict['M_action']
        qf1, qf2 = self.critic(M_obs, M_action)
        qf1 = qf1.reshape(self.M*batch_size)
        qf2 = qf2.reshape(self.M*batch_size)
        next_q_value = next_q_value.repeat(self.M)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        with torch.no_grad():
            td_error = (0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value))).reshape(self.M, batch_size).mean(dim=0)
        return qf1_loss, qf2_loss, td_error
