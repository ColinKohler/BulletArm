from bulletarm_baselines.equi_rl.agents.base_agent import BaseAgent
import numpy as np
import torch
from copy import deepcopy

class DQNBase(BaseAgent):
    """
    Base class for DQN agent
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr)

        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # discrete action spaces
        self.p_range = torch.tensor([1])
        if n_p == 2:
            self.p_range = torch.tensor([0, 1])

        self.d_theta_range = torch.tensor([0])
        if n_theta == 3:
            self.d_theta_range = torch.tensor([-dr, 0, dr])

        self.dxy_range = torch.tensor([[-dx, -dy], [-dx, 0], [-dx, dy],
                                       [0, -dy], [0, 0], [0, dy],
                                       [dx, -dy], [dx, 0], [dx, dy]])
        self.dz_range = torch.tensor([-dz, 0, dz])

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = 1e-2

        for t_param, l_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def updateTarget(self):
        """
        Disable the default hard target update
        """
        pass

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        """
        Forward pass the Q-network
        :param state: gripper state
        :param obs: observation
        :param target_net: whether to use the target network
        :param to_cpu: move output to cpu
        :return: the output of the Q-network
        """
        raise NotImplementedError

    def calcTDLoss(self):
        """
        Calculate the TD loss
        :return: TD loss
        """
        raise NotImplementedError

    def initNetwork(self, network, initialize_target=True):
        """
        Initialize networks
        :param network: Q-network
        :param initialize_target: whether to initialize the target network
        """
        self.policy_net = network
        if initialize_target:
            self.target_net = deepcopy(network)
            self.target_networks.append(self.target_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.networks.append(self.policy_net)
        self.optimizers.append(self.optimizer)

    def decodeActions(self, p_id, dxy_id, dz_id, dtheta_id):
        """
        Get actions from the action ids
        :param p_id: gripper primitive id
        :param dxy_id: delta xy id
        :param dz_id: delta z id
        :param dtheta_id: delta theta id
        :return: action ids, actions
        """
        p = self.p_range[p_id]
        dxy = self.dxy_range[dxy_id]
        dz = self.dz_range[dz_id]
        dtheta = self.d_theta_range[dtheta_id]
        actions = torch.stack([p, dxy[:, 0], dxy[:, 1], dz, dtheta], dim=1)
        action_idxes = torch.stack([p_id, dxy_id, dz_id, dtheta_id], dim=1)
        return action_idxes, actions

    def getActionFromPlan(self, plan):
        """
        Get action ids and actions from planner actions
        :param plan: planner actions
        :return: action ids, actions
        """
        primitive = plan[:, 0:1]
        dxy = plan[:, 1:3]
        dz = plan[:, 3:4]
        dr = plan[:, 4:5]

        p_id = torch.argmin(torch.abs(self.p_range - primitive), 1)
        dxy_id = torch.argmin((dxy.unsqueeze(1) - self.dxy_range).abs().sum(2), 1)
        dz_id = torch.argmin(torch.abs(self.dz_range - dz), 1)
        dtheta_id = torch.argmin(torch.abs(self.d_theta_range - dr), 1)

        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: td loss, td error
        """
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error