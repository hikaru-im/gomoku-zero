import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResNetGomoku(nn.Module):
    """Dual-head residual network for 15x15 Gomoku (AlphaZero-style).

    Input:  (batch, 3, 15, 15) — [current stones, opponent stones, color to move]
    Output: policy logits (batch, 225), value scalar (batch, 1)
    """

    def __init__(self, in_channels=3, board_size=15, num_blocks=10, num_filters=128):
        super().__init__()
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.num_filters = num_filters

        # Input tower
        self.conv_input = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Tower
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, x):
        """Inference mode: returns (log_probs, value).
        log_probs: (batch, 225) log-softmax over moves.
        value: (batch,) scalar in [-1, 1].
        """
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(x)
            log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, value.squeeze(-1)

    def loss(self, states, target_policies, target_values):
        """Compute combined loss = MSE(value) + CrossEntropy(policy).

        L2 regularization is handled by the optimizer (weight_decay).

        Args:
            states: (batch, 3, 15, 15) float32
            target_policies: (batch, 225) float32 — ground truth MCTS visit distribution
            target_values: (batch,) float32 — outcome z ∈ {-1, 0, 1}

        Returns:
            total_loss (scalar), policy_loss, value_loss
        """
        logits, value = self.forward(states)
        value = value.squeeze(-1)

        # Cross-entropy with soft targets (probability distributions)
        log_p = F.log_softmax(logits, dim=-1).clamp(min=-100)
        policy_loss = -(target_policies * log_p).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, target_values)

        return policy_loss + value_loss, policy_loss, value_loss

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.state_dict(),
            "num_blocks": self.num_blocks,
            "num_filters": self.num_filters,
        }, path)

    def load_checkpoint(self, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(ckpt["model_state_dict"])
        return self
