import torch.nn as nn


class PlayerPosition(nn.Module):
    def __init__(self, input_size):
        super(PlayerPosition, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
        )

    def forward(self, x):
        position = self.linear_relu_stack(x)
        return position
