import numpy as np
from bulletarm_baselines.fc_dqn.agents.agents_3d.dqn_3d_asr import DQN3DASR
from bulletarm_baselines.fc_dqn.agents.margin_base import MarginBase

class Margin3DASR(DQN3DASR, MarginBase):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), margin='l', margin_l=0.1, margin_weight=0.1,
                 softmax_beta=100):
        DQN3DASR.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        MarginBase.__init__(self, margin, margin_l, margin_weight, softmax_beta)

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        q1_output = self.loss_calc_dict['q1_output']
        q2_output = self.loss_calc_dict['q2_output']
        action_idx_dense = action_idx[:, 0] * self.heightmap_size + action_idx[:, 1]
        q1_margin_loss = self.getMarginLossSingle(q1_output.reshape(q1_output.size(0), -1), action_idx_dense, is_experts, True)
        q2_margin_loss = self.getMarginLossSingle(q2_output, action_idx[:, 2], is_experts)
        return q1_margin_loss + q2_margin_loss

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        margin_loss = self.calcMarginLoss()
        loss = td_loss + self.margin_weight * margin_loss

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        self.loss_calc_dict = {}

        return loss.item(), td_error