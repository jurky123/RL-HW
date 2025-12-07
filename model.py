import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()
        planes = 128
        self.conv_in = nn.Sequential(
            nn.Conv2d(6, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        self.res_layers = nn.Sequential(
            *[self._make_block(planes) for _ in range(10)]  # 10 blocks
        )

        flat_dim = planes * 4 * 9

        self.policy = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 235)
        )

        self.value = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def _make_block(self, planes):
        return nn.Sequential(
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        obs = x["observation"].float()
        mask = x["action_mask"].float()

        out = self.conv_in(obs)
        out = self.res_layers(out)
        out = out.flatten(1)

        logits = self.policy(out)
        value = self.value(out)

        logits += torch.clamp(torch.log(mask), -1e38, 1e38)
        return logits, value