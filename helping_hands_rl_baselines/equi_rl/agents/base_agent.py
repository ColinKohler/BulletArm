import numpy as np
import torch
from helping_hands_rl_baselines.equi_rl.utils.torch_utils import augmentTransition
from itertools import repeat
from helping_hands_rl_baselines.equi_rl.utils.parameters import obs_type

class BaseAgent:
    """
    The base RL agent class
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        # magnitude of actions
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dr = dr

        self.networks = []
        self.target_networks = []
        self.optimizers = []

        self.loss_calc_dict = {}

        self.aug = False
        self.aug_type = 'se2'

    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: loss
        """
        raise NotImplementedError

    def getEGreedyActions(self, state, obs, eps):
        """
        Get e-greedy actions
        :param state: gripper holding state
        :param obs: observation
        :param eps: epsilon
        :return: action
        """
        raise NotImplementedError

    def getGreedyActions(self, state, obs):
        """
        Get greedy actions
        :param state: gripper holding state
        :param obs: observation
        :return: action
        """
        return self.getEGreedyActions(state, obs, 0)

    def _loadBatchToDevice(self, batch):
        """
        Load batch into pytorch tensor
        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        """
        if self.aug:
            # perform augmentation for RAD
            batch = list(map(augmentTransition, batch, repeat(self.aug_type)))

        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.tensor(np.stack(states)).long().to(self.device)
        obs_tensor = torch.tensor(np.stack(images)).to(self.device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack(xys)).to(self.device)
        rewards_tensor = torch.tensor(np.stack(rewards)).to(self.device)
        next_states_tensor = torch.tensor(np.stack(next_states)).long().to(self.device)
        next_obs_tensor = torch.tensor(np.stack(next_obs)).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack(dones)).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.tensor(np.stack(step_lefts)).to(self.device)
        is_experts_tensor = torch.tensor(np.stack(is_experts)).bool().to(self.device)

        if obs_type is 'pixel':
            # scale observation from int to float
            obs_tensor = obs_tensor/255*0.4
            next_obs_tensor = next_obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def _loadLossCalcDict(self):
        """
        Get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def train(self):
        """
        Call .train() for all models
        """
        for i in range(len(self.networks)):
            self.networks[i].train()
        for i in range(len(self.target_networks)):
            self.target_networks[i].train()

    def eval(self):
        """
        Call .eval() for all models
        """
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def getModelStr(self):
        """
        Get the str of all models (for logging)
        :return: str of all models
        """
        return str(self.networks)

    def updateTarget(self):
        """
        Hard update the target networks
        """
        for i in range(len(self.networks)):
            self.target_networks[i].load_state_dict(self.networks[i].state_dict())

    def loadModel(self, path_pre):
        """
        Load the saved models
        :param path_pre: path prefix to the model
        """
        for i in range(len(self.networks)):
            path = path_pre + '_{}.pt'.format(i)
            print('loading {}'.format(path))
            self.networks[i].load_state_dict(torch.load(path))
        self.updateTarget()

    def saveModel(self, path_pre):
        """
        Save the models with path prefix path_pre. a '_q{}.pt' suffix will be added to each model
        :param path_pre: path prefix
        """
        for i in range(len(self.networks)):
            torch.save(self.networks[i].state_dict(), '{}_{}.pt'.format(path_pre, i))

    def getSaveState(self):
        """
        Get the save state for checkpointing. Include network states, target network states, and optimizer states
        :return: the saving state dictionary
        """
        state = {}
        for i in range(len(self.networks)):
            self.networks[i].to('cpu')
            state['{}'.format(i)] = self.networks[i].state_dict()
            state['{}_optimizer'.format(i)] = self.optimizers[i].state_dict()
            self.networks[i].to(self.device)
        for i in range(len(self.target_networks)):
            self.target_networks[i].to('cpu')
            state['{}_target'.format(i)] = self.target_networks[i].state_dict()
            self.target_networks[i].to(self.device)
        return state

    def loadFromState(self, save_state):
        """
        Load from a save_state
        :param save_state: the loading state dictionary
        """
        for i in range(len(self.networks)):
            self.networks[i].to('cpu')
            self.networks[i].load_state_dict(save_state['{}'.format(i)])
            self.networks[i].to(self.device)
            self.optimizers[i].load_state_dict(save_state['{}_optimizer'.format(i)])
        for i in range(len(self.target_networks)):
            self.target_networks[i].to('cpu')
            self.target_networks[i].load_state_dict(save_state['{}_target'.format(i)])
            self.target_networks[i].to(self.device)

    def copyNetworksFrom(self, from_agent):
        """
        Copy networks from another agent
        :param from_agent: the agent being copied from
        """
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(from_agent.networks[i].state_dict())
