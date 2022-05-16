import numpy as np
import torch
from bulletarm_baselines.fc_dqn.agents.base_agent import BaseAgent
from bulletarm_baselines.fc_dqn.utils.torch_utils import perturb, ExpertTransition


class Base3D(BaseAgent):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)

        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()

        self.aug = False
        self.aug_type = 'se2'

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rot = plan[:, 2:3]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)
        diff = (rot.expand(-1, self.num_rz) - self.rzs).abs()
        diff2 = (diff - np.pi).abs()
        rot_id = torch.min(diff, diff2).argmin(1).unsqueeze(1)
        # rot_id = (rot.expand(-1, self.num_rz) - self.rzs).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rot = self.rzs[rot_id]
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rot_id), dim=1)
        return action_idx, actions

    def augmentTransitionCn(self, d):
        dtheta = self.rzs[1] - self.rzs[0]
        theta_dis_n = 2 * np.pi // dtheta
        obs, next_obs, _, (trans_pixel,), transform_params = perturb(d.obs[0].clone().numpy(),
                                                                     d.next_obs[0].clone().numpy(),
                                                                     [d.action[:2].clone().numpy()],
                                                                     set_trans_zero=True,
                                                                     theta_dis_n=theta_dis_n)
        action_theta = d.action[2].clone()
        trans_theta, _, _ = transform_params
        if trans_theta >= dtheta:
            action_theta -= (trans_theta // dtheta).long()
            action_theta %= self.num_rz
        if trans_theta <= -dtheta:
            action_theta += (trans_theta // dtheta).long()
            action_theta %= self.num_rz
        trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
        return ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert)

    def augmentTransitionSE2(self, d):
        dtheta = self.rzs[1] - self.rzs[0]
        theta_dis_n = 2*np.pi // dtheta
        obs, next_obs, _, (trans_pixel,), transform_params = perturb(d.obs[0].clone().numpy(), d.next_obs[0].clone().numpy(), [d.action[:2].clone().numpy()], theta_dis_n=theta_dis_n)
        action_theta = d.action[2].clone()
        trans_theta, _, _ = transform_params
        if trans_theta >= dtheta:
            action_theta -= (trans_theta // dtheta).long()
            action_theta %= self.num_rz
        if trans_theta <= -dtheta:
            action_theta += (trans_theta // dtheta).long()
            action_theta %= self.num_rz
        trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
        return ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert)

    def augmentTransitionTranslate(self, d):
        obs, next_obs, _, (trans_pixel,), transform_params = perturb(d.obs[0].clone().numpy(), d.next_obs[0].clone().numpy(), [d.action[:2].clone().numpy()], set_theta_zero=True)
        trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], d.action[2]])
        return ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert)


    def augmentTransition(self, d):
        if self.aug_type == 'se2':
            return self.augmentTransitionSE2(d)
        elif self.aug_type == 'cn':
            return self.augmentTransitionCn(d)
        elif self.aug_type == 't':
            return self.augmentTransitionTranslate(d)
        else:
            raise NotImplementedError

    def _loadBatchToDevice(self, batch):
        if self.aug:
            batch = list(map(self.augmentTransition, batch))
        super()._loadBatchToDevice(batch)
