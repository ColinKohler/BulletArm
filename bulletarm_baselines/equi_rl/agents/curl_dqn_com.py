from bulletarm_baselines.equi_rl.agents.dqn_agent_com import DQNAgentCom
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.equi_rl.utils.torch_utils import randomCrop, centerCrop
import torch.nn as nn

class CURLDQNCom(DQNAgentCom):
    """
    CURL DQN agent class
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1, crop_size=128):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.momentum_net = None
        # the coefficient of contrastive loss
        # self.coeff = 1
        self.coeff = 0.01
        self.crop_size = crop_size

    def initialize_momentum_net(self):
        for param_q, param_k in zip(self.policy_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

    # Code for this function from https://github.com/facebookresearch/moco
    @torch.no_grad()
    def update_momentum_net(self, momentum=0.999):
        for param_q, param_k in zip(self.policy_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)  # update

    def initNetwork(self, network, initialize_target=True):
        self.policy_net = network
        self.target_net = deepcopy(network)
        self.momentum_net = deepcopy(network)
        self.initialize_momentum_net()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.networks.append(self.policy_net)

        self.target_networks.append(self.target_net)
        self.target_networks.append(self.momentum_net)

        self.optimizers.append(self.optimizer)

        for param in self.target_net.parameters():
            param.requires_grad = False

        for param in self.momentum_net.parameters():
            param.requires_grad = False

    def getEGreedyActions(self, state, obs, eps):
        obs = centerCrop(obs, out=self.crop_size)
        return super().getEGreedyActions(state, obs, eps)

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q, h = net(stacked.to(self.device))
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_xy, self.n_z, self.n_theta, self.n_p)
        return q

    def forwardPolicyNetWithH(self, state, obs, to_cpu=False):
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q, h = self.policy_net(stacked.to(self.device))
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_xy, self.n_z, self.n_theta, self.n_p)
        return q, h

    def forwardMomentumNetWithH(self, state, obs, to_cpu=False):
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q, h = self.momentum_net(stacked.to(self.device))
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_xy, self.n_z, self.n_theta, self.n_p)
        return q, h

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()

        # aug_obs_1 = aug(obs)
        # aug_obs_2 = aug(obs)

        aug_obs_1 = randomCrop(obs, out=self.crop_size)
        aug_obs_2 = randomCrop(obs, out=self.crop_size)
        obs = centerCrop(obs, out=self.crop_size)
        next_obs = centerCrop(next_obs, out=self.crop_size)

        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_all_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_all_prime.reshape(batch_size, -1).max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q = self.forwardNetwork(states, obs)

        _, z_anch = self.forwardPolicyNetWithH(states, aug_obs_1)
        _, z_target = self.forwardMomentumNetWithH(states, aug_obs_2)
        z_proj = torch.matmul(self.policy_net.W, z_target.T)
        logits = torch.matmul(z_anch, z_proj)
        logits = (logits - torch.max(logits, 1)[0][:, None])
        logits = logits * 0.1
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        moco_loss = (nn.CrossEntropyLoss()(logits, labels)).to(self.device)

        q_pred = q[torch.arange(batch_size), dxy_id, dz_id, dtheta_id, p_id]
        self.loss_calc_dict['q_output'] = q
        self.loss_calc_dict['q_pred'] = q_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)

        loss = td_loss + (moco_loss * self.coeff)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.targetSoftUpdate()
        self.update_momentum_net()

        self.loss_calc_dict = {}

        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)

        return (td_loss.item(), moco_loss.item()), td_error