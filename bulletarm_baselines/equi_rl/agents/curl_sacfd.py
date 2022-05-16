from bulletarm_baselines.equi_rl.agents.curl_sac import CURLSAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.equi_rl.networks.curl_sac_net import CURL
from bulletarm_baselines.equi_rl.utils.torch_utils import randomCrop, centerCrop

class CURLSACfD(CURLSAC):
    """
    CURL (FERM) SACfD agent class
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi / 16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False, z_dim=50,
                 crop_size=64, demon_w=0.1, demon_l='pi'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, z_dim, crop_size)
        self.demon_w = demon_w
        assert demon_l in ['mean', 'pi']
        self.demon_l = demon_l

    def update(self, batch):
        self._loadBatchToDevice(batch)

        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()

        next_obs = randomCrop(next_obs, out=self.crop_size)
        obs = randomCrop(obs, out=self.crop_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_obs)
            next_state_log_pi = next_state_log_pi.reshape(batch_size)
            qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action)
            qf1_next_target = qf1_next_target.reshape(batch_size)
            qf2_next_target = qf2_next_target.reshape(batch_size)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + non_final_masks * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(obs, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.reshape(batch_size)
        qf2 = qf2.reshape(batch_size)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, mean = self.actor.sample(obs, detach_encoder=True)
        qf1_pi, qf2_pi = self.critic(obs, pi, detach_encoder=True)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # add expert loss
        if is_experts.sum():
            if self.demon_l == 'pi':
                demon_loss = F.mse_loss(pi[is_experts], action[is_experts])
                policy_loss += self.demon_w * demon_loss
            else:
                demon_loss = F.mse_loss(mean[is_experts], action[is_experts])
                policy_loss += self.demon_w * demon_loss


        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        curl_loss = self.updateCURL(update_target=False)

        with torch.no_grad():
            td_error = 0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value))

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), curl_loss.item(), alpha_tlogs.item()), td_error
