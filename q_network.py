import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_shape, action_space_size, seed):
        super(QNetwork, self).__init__()
        state_shape = state_shape[0]*state_shape[1]
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the 2D matrix input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x