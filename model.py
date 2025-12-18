import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """稳健残差块，支持梯度稳定训练"""
    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)
        self.relu = nn.ReLU(inplace=True)
        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        # 防止数值异常
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 降低通道数，从 128 降到 64
        planes = 64 

        # stem
        self.conv_in = nn.Sequential(
            nn.Conv2d(10, planes, 3, padding=1, bias=False),
            nn.GroupNorm(4, planes), # 减少 group 数
            nn.ReLU(inplace=True),
        )

        # 2. 大幅减少残差块，从 16 降到 3 或 4
        self.res_layers = nn.Sequential(
            *[ResidualBlock(planes) for _ in range(4)] 
        )

        # 3. 简化 Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(planes, 16, 1, bias=False), # 通道进一步压缩
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 4 * 9, 256), # 1024 降到 256
            nn.ReLU(),
            nn.Linear(256, 235),
        )

        # 4. 简化 Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(planes, 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 4 * 9, 128), # 512 降到 128
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        obs = x["observation"].float()
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        out = self.conv_in(obs)
        out = self.res_layers(out)

        logits = self.policy_head(out)
        value = self.value_head(out)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)

        return logits, value
