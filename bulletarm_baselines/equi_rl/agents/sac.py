from bulletarm_baselines.equi_rl.agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.equi_rl.utils.parameters import heightmap_size, crop_size
from bulletarm_baselines.equi_rl.utils.torch_utils import centerCrop

class SAC(A2CBase):
    """
    SAC agent class
    Part of the code for this class is referenced from https://github.com/pranz24/pytorch-soft-actor-critic
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.obs_type = obs_type

        if self.policy_type == 'gaussian':
            if self.automatic_entropy_tuning is True:
                # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.target_entropy = -n_a
                self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
                # self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=(0.5, 0.999))
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-3)

        self.num_update = 0

    def initNetwork(self, actor, critic, initialize_target=True):
        """
        Initialize networks
        :param actor: actor network
        :param critic: critic network
        :param initialize_target: whether to create target networks
        """
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])
        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)
        self.optimizers.append(self.alpha_optim)

    def getSaveState(self):
        """
        Get the save state for checkpointing. Include network states, target network states, and optimizer states
        :return: the saving state dictionary
        """
        state = super().getSaveState()
        state['alpha'] = self.alpha
        state['log_alpha'] = self.log_alpha
        state['alpha_optimizer'] = self.alpha_optim.state_dict()
        return state

    def loadFromState(self, save_state):
        """
        Load from a save_state
        :param save_state: the loading state dictionary
        """
        super().loadFromState(save_state)
        self.alpha = save_state['alpha']
        self.log_alpha = torch.tensor(np.log(self.alpha.item()), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-3)
        self.alpha_optim.load_state_dict(save_state['alpha_optimizer'])

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


    def getEGreedyActions(self, state, obs, eps):
        """
        Get stochastic behavior policy's action. Note that this function is called getEGreedyActions, but it uses SAC's
        gaussian distribution to sample actions instead of e-greedy
        :param state: gripper holding state
        :param obs: observation
        :param eps: epsilon (not used)
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        return self.getSACAction(state, obs, evaluate=False)

    def getGreedyActions(self, state, obs):
        """
        Get greedy actions
        :param state: gripper holding state
        :param obs: observation
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        return self.getSACAction(state, obs, evaluate=True)

    def getSACAction(self, state, obs, evaluate):
        """
        Get SAC action (greedy or sampled from gaussian, based on evaluate flag)
        :param state: gripper holding state
        :param obs: observation
        :param evaluate: if evaluate==True, return greedy action. Otherwise return action sampled from gaussian
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        with torch.no_grad():
            if self.obs_type is 'pixel':
                state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
                obs = torch.cat([obs, state_tile], dim=1).to(self.device)
                if heightmap_size > crop_size:
                    obs = centerCrop(obs, out=crop_size)
            else:
                obs = obs.to(self.device)

            if evaluate is False:
                action, _, _ = self.actor.sample(obs)
            else:
                _, _, action = self.actor.sample(obs)
            action = action.to('cpu')
            return self.decodeActions(*[action[:, i] for i in range(self.n_a)])

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = super()._loadLossCalcDict()

        if self.obs_type is 'pixel':
            # stack state as the second channel of the obs
            obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
            next_obs = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)

        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def calcActorLoss(self):
        """
        Calculate actor loss
        :return: actor loss
        """
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pi, log_pi, mean = self.actor.sample(obs)
        self.loss_calc_dict['pi'] = pi
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi

        qf1_pi, qf2_pi = self.critic(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        return policy_loss

    def calcCriticLoss(self):
        """
        Calculate critic loss
        :return: q1 loss, q2 loss, td error
        """
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
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
        with torch.no_grad():
            td_error = 0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value))
        return qf1_loss, qf2_loss, td_error

    def updateActorAndAlpha(self):
        """
        Update actor and alpha
        :return: policy_loss, alpha_loss, alpha
        """
        policy_loss = self.calcActorLoss()
        log_pi = self.loss_calc_dict['log_pi']

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return policy_loss, alpha_loss, alpha_tlogs

    def updateCritic(self):
        """
        Update critic
        :return: q1 loss, q2 loss, td error
        """
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        return qf1_loss, qf2_loss, td_error

    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: loss
        """
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, td_error = self.updateCritic()
        policy_loss, alpha_loss, alpha_tlogs = self.updateActorAndAlpha()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()), td_error
