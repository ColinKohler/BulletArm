from bulletarm_baselines.fc_dqn.agents.agents_2d.dqn_2d_fcn import DQN2DFCN
from bulletarm_baselines.fc_dqn.agents.margin_base import MarginBase

class Margin2DFCN(DQN2DFCN, MarginBase):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, margin='l', margin_l=0.1, margin_weight=0.1, softmax_beta=100):
        DQN2DFCN.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)
        MarginBase.__init__(self, margin, margin_l, margin_weight, softmax_beta)

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        q1_output = self.loss_calc_dict['q1_output']
        return self.getMarginLossSingle(q1_output.reshape(q1_output.size(0), -1), action_idx[:, 0] * self.heightmap_size + action_idx[:, 1], is_experts, True)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()
        margin_loss = self.calcMarginLoss()
        loss = td_loss + self.margin_weight * margin_loss

        self.fcn_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return loss.item(), td_error