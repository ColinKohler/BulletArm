import torch
import torch.nn.functional as F

class MarginBase:
    def __init__(self, margin='l', margin_l=0.1, margin_weight=0.1, softmax_beta=100):
        self.margin = margin
        self.margin_l = margin_l
        self.margin_weight = margin_weight
        self.softmax_beta = softmax_beta

    def getMarginLossSingle(self, q_output, action_idx, is_experts, apply_beta=False):
        if is_experts.sum() == 0:
            return torch.tensor(0)
        batch_size = q_output.size(0)
        q_pred = q_output[torch.arange(0, batch_size), action_idx]

        if self.margin == 'ce':
            if apply_beta:
                margin_loss = F.cross_entropy(self.softmax_beta*q_output[is_experts], action_idx[is_experts])
            else:
                margin_loss = F.cross_entropy(q_output[is_experts], action_idx[is_experts])

        elif self.margin == 'oril':
            margin = torch.ones_like(q_output) * self.margin_l
            margin[torch.arange(0, batch_size), action_idx] = 0
            margin_output = q_output + margin
            margin_output_max = margin_output.reshape(batch_size, -1).max(1)[0]
            margin_loss = (margin_output_max - q_pred)[is_experts]
            margin_loss = margin_loss.mean()

        elif self.margin == 'l':
            margin_losses = []
            for j in range(batch_size):
                if not is_experts[j]:
                    margin_losses.append(torch.tensor(0).float().to(q_output.device))
                    continue

                qe = q_pred[j]
                q_all = q_output[j]
                over = q_all[(q_all > qe - self.margin_l) * (torch.arange(0, q_all.shape[0]).to(q_output.device)!=action_idx[j])]
                if over.shape[0] == 0:
                    margin_losses.append(torch.tensor(0).float().to(q_output.device))
                else:
                    over_target = torch.ones_like(over) * qe - self.margin_l
                    margin_losses.append((over - over_target).mean())
            margin_loss = torch.stack(margin_losses).mean()

        else:
            raise NotImplementedError

        return margin_loss
