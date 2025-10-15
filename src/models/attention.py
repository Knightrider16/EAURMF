import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, pool_type='avg'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.pool_type = pool_type
        
        # Conv1d for attention over channels
        self.conv = nn.Conv1d(
            in_channels=gate_channels,
            out_channels=gate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):
        # x: [B, C, L]
        if self.pool_type == 'avg':
            pooled = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        elif self.pool_type == 'max':
            pooled = x.max(dim=2, keepdim=True).values
        elif self.pool_type == 'lp':
            pooled = x.norm(2, dim=2, keepdim=True)
        elif self.pool_type == 'lse':
            pooled = torch.logsumexp(x, dim=2, keepdim=True)
        else:
            raise ValueError("Invalid pool type")

        # Channel attention via conv
        att = self.conv(pooled)  # [B, C, 1]
        score = torch.sigmoid(att)  # normalize scores between 0–1

        # Apply attention and residual fusion
        x = x * score + x  # [B, C, L]

        # Optional fusion: flatten or reduce across channels
        fusion_output = torch.sum(x, dim=1)  # [B, L]
        return fusion_output


if __name__ == "__main__":
    a = torch.randn(50, 3, 72)
    fusion = ChannelGate(3, reduction_ratio=3, pool_type='avg')
    b = fusion(a)
    print(b.shape)  # should print torch.Size([50, 72])
