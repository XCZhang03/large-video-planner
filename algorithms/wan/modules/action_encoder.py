import torch
from torch import nn
import torch.nn.functional as F



class CausalConv1D(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        # x: (B, D, T)
        pad_left = self.kernel_size - 1
        x = F.pad(x, (pad_left, 0))  # causal padding
        return self.conv(x)

class ActionEncoder(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        # this conv collapses 4 timesteps → 1
        self.temporal_down = CausalConv1D(
            hidden_dim,
            kernel_size=3,
            stride=4
        )
        self.proj = nn.Sequential(
            torch.nn.Linear(hidden_dim, 6*hidden_dim),
            torch.nn.SiLU(),
        )

    def encode_chunk(self, x):
        """
        x: (B, T_chunk, D)
        returns: (B, 1, D)
        """
        x = x.transpose(1, 2)      # (B, D, T)
        x = self.temporal_down(x)  # (B, D, 1)
        x = x.transpose(1, 2)      # (B, 1, D)
        return x

    def forward(self, x):
        """
        x: (B, 1+T, D)
        returns: (B, 1+T//4, D)
        """
        B, T, D = x.shape
        x = self.encode(x)  # (B, T, D)
        iters = 1 + (T - 1) // 4 if T > 0 else 1

        outputs = []
        for i in range(iters):
            if i == 0:
                chunk = x[:, :1, :]                # first timestep
            else:
                chunk = x[:, 1 + 4*(i-1) : 1 + 4*i, :]

            out = self.encode_chunk(chunk)
            outputs.append(out)
        x = torch.cat(outputs, dim=1)  # (B, 1+T//4, D)
        x_ = self.proj(x).unflatten(-1, (6, self.hidden_dim))
        return x, x_
