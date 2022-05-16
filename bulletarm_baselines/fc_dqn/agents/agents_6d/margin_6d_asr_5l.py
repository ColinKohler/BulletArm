import numpy as np
from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L
from bulletarm_baselines.fc_dqn.agents.margin_base import MarginBase

class Margin6DASR5L(DQN6DASR5L, MarginBase):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), num_ry=8, ry_range=(0, 7*np.pi/8), num_rx=8,
                 rx_range=(0, 7*np.pi/8), num_zs=16, z_range=(0.02, 0.12), margin='l', margin_l=0.1, margin_weight=0.1,
                 softmax_beta=100):
        DQN6DASR5L.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz,
                            rz_range, num_ry, ry_range, num_rx, rx_range, num_zs, z_range)
        MarginBase.__init__(self, margin, margin_l, margin_weight, softmax_beta)

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        q1_output = self.loss_calc_dict['q1_output']
        q2_output = self.loss_calc_dict['q2_output']
        q3_output = self.loss_calc_dict['q3_output']
        q4_output = self.loss_calc_dict['q4_output']
        q5_output = self.loss_calc_dict['q5_output']
        a1_idx_dense = action_idx[:, 0] * self.heightmap_size + action_idx[:, 1]
        q1_margin_loss = self.getMarginLossSingle(q1_output.reshape(q1_output.size(0), -1), a1_idx_dense, is_experts, True)
        q2_margin_loss = self.getMarginLossSingle(q2_output, action_idx[:, 3], is_experts)
        q3_margin_loss = self.getMarginLossSingle(q3_output, action_idx[:, 2], is_experts)
        q4_margin_loss = self.getMarginLossSingle(q4_output, action_idx[:, 4], is_experts)
        q5_margin_loss = self.getMarginLossSingle(q5_output, action_idx[:, 5], is_experts)
        return q1_margin_loss + q2_margin_loss + q3_margin_loss + q4_margin_loss + q5_margin_loss

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        margin_loss = self.calcMarginLoss()
        loss = td_loss + self.margin_weight * margin_loss

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.q3_optimizer.zero_grad()
        self.q4_optimizer.zero_grad()
        self.q5_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        for param in self.q3.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q3_optimizer.step()

        for param in self.q4.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q4_optimizer.step()

        for param in self.q5.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q5_optimizer.step()

        self.loss_calc_dict = {}

        return loss.item(), td_error