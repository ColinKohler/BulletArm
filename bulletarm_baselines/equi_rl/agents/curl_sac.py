from bulletarm_baselines.equi_rl.agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.equi_rl.networks.curl_sac_net import CURL
from bulletarm_baselines.equi_rl.utils.torch_utils import randomCrop, centerCrop

class CURLSAC(SAC):
    """
    CURL SAC (FERM) agent class
    Part of the code for this class is referenced from: https://github.com/PhilipZRH/ferm
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi / 16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False, z_dim=50, crop_size=64):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval, automatic_entropy_tuning)
        self.z_dim = z_dim
        self.crop_size = crop_size
        self.encoder_optimizer = None
        self.curl = None
        self.curl_optimizer = None

        self.encoder_tau = 0.05

    def encoderTargetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        for t_param, l_param in zip(
                self.critic_target.encoder.parameters(), self.critic.encoder.parameters()
        ):
            t_param.data.copy_(self.encoder_tau * l_param.data + (1.0 - self.encoder_tau) * t_param.data)

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        for t_param, l_param in zip(
                self.critic_target.q1.parameters(), self.critic.q1.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.q2.parameters(), self.critic.q2.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

        self.encoderTargetSoftUpdate()

    def initNetwork(self, actor, critic, initialize_target=True):
        """
        Initialize networks
        :param actor: actor network
        :param critic: critic network
        :param initialize_target: whether to create target networks
        """
        self.actor = actor
        self.critic = critic
        self.critic_target = deepcopy(critic)

        self.actor.encoder.copyConvWeightsFrom(self.critic.encoder)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=self.lr[2])

        self.curl = CURL(self.z_dim, self.critic.encoder, self.critic_target.encoder).to(self.device)
        self.curl_optimizer = torch.optim.Adam(self.curl.parameters(), lr=self.lr[3])

        self.target_networks.append(self.critic_target)

        self.networks.append(self.curl)
        self.networks.append(self.actor)
        self.networks.append(self.critic)

        self.optimizers.append(self.encoder_optimizer)
        self.optimizers.append(self.curl_optimizer)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

    def loadFromState(self, save_state):
        """
        Load from a save_state
        :param save_state: the loading state dictionary
        """
        super().loadFromState(save_state)
        self.actor.encoder.copyConvWeightsFrom(self.critic.encoder)

    def loadModel(self, path_pre):
        """
        Load the saved models
        :param path_pre: path prefix to the model
        """
        super().loadModel(path_pre)
        self.actor.encoder.copyConvWeightsFrom(self.critic.encoder)

    def updateCURL(self, update_target=False):
        """
        Update CURL
        :param update_target: whether to update target encoder
        :return: curl loss
        """
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        obs_anchor = randomCrop(obs, out=self.crop_size)
        obs_pos = randomCrop(obs, out=self.crop_size)

        z_a = self.curl.encode(obs_anchor)
        z_pos = self.curl.encode(obs_pos, ema=True)

        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = F.cross_entropy(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.curl_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.curl_optimizer.step()

        if update_target:
            self.encoderTargetSoftUpdate()

        return loss

    def getSACAction(self, state, obs, evaluate):
        """
        Get SAC action (greedy or sampled from gaussian, based on evaluate flag)
        :param state: gripper holding state
        :param obs: observation
        :param evaluate: if evaluate==True, return greedy action. Otherwise return action sampled from gaussian
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        with torch.no_grad():
            state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
            stacked = torch.cat([obs, state_tile], dim=1).to(self.device)

            stacked = centerCrop(stacked, out=self.crop_size)

            if evaluate is False:
                action, _, _ = self.actor.sample(stacked)
            else:
                _, _, action = self.actor.sample(stacked)
            action = action.to('cpu')
            return self.decodeActions(*[action[:, i] for i in range(self.n_a)])

    def updateCURLOnly(self, batch):
        """
        CURL pre-training update function
        :param batch: the sampled minibatch
        :return: loss
        """
        self._loadBatchToDevice(batch)
        curl_loss = self.updateCURL(update_target=True)
        return 0, 0, 0, 0, curl_loss.item(), 0


    def update(self, batch):
        """
        CURL normal training step
        :param batch: the sampled minibatch
        :return: loss
        """
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

        pi, log_pi, _ = self.actor.sample(obs, detach_encoder=True)
        qf1_pi, qf2_pi = self.critic(obs, pi, detach_encoder=True)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

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
