import torch
from bulletarm import env_factory

class EnvWrapper:
    def __init__(self, num_processes, env, env_config, planner_config):
        self.envs = env_factory.createEnvs(num_processes, env, env_config, planner_config)

    def reset(self):
        (states, in_hands, obs) = self.envs.reset()
        states = torch.tensor(states).float()
        obs = torch.tensor(obs).float()
        return states, obs

    def getNextAction(self):
        return torch.tensor(self.envs.getNextAction()).float()

    def step(self, actions, auto_reset=False):
        actions = actions.cpu().numpy()
        (states_, in_hands_, obs_), rewards, dones = self.envs.step(actions, auto_reset)
        states_ = torch.tensor(states_).float()
        obs_ = torch.tensor(obs_).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        return states_, obs_, rewards, dones

    def stepAsync(self, actions, auto_reset=False):
        actions = actions.cpu().numpy()
        self.envs.stepAsync(actions, auto_reset)

    def stepWait(self):
        (states_, in_hands_, obs_), rewards, dones = self.envs.stepWait()
        states_ = torch.tensor(states_).float()
        obs_ = torch.tensor(obs_).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        return states_, obs_, rewards, dones

    def getStepLeft(self):
        return torch.tensor(self.envs.getStepsLeft()).float()

    def reset_envs(self, env_nums):
        states, in_hands, obs = self.envs.reset_envs(env_nums)
        states = torch.tensor(states).float()
        obs = torch.tensor(obs).float()
        return states, obs

    def close(self):
        self.envs.close()

    def saveToFile(self, envs_save_path):
        return self.envs.saveToFile(envs_save_path)

    def getEnvGitHash(self):
        return self.envs.getEnvGitHash()

    def getEmptyInHand(self):
        return torch.tensor(self.envs.getEmptyInHand()).float()