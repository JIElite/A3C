import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Network, self).__init__()

        self.l1 = nn.Linear(n_features, 64)
        self.l2 = nn.Linear(64, 64)

        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        value = self.value_head(x)
        soft_prob = F.softmax(self.policy_head(x), dim=-1)
        return soft_prob, value

    def get_value(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.value_head(x)


class SeparateNetwork(nn.Module):
    def __init__(self, n_features, n_actions):
        super(SeparateNetwork, self).__init__()
        self.p1 = nn.Linear(n_features, 64)
        self.p2 = nn.Linear(64, n_actions)
        self.v1 = nn.Linear(n_features, 64)
        self.v2 = nn.Linear(64, 1)
    
    def forward(self, x):
        p = F.relu(self.p1(x))
        p = F.softmax(self.p2(p), dim=-1)
        v = F.relu(self.v1(x))
        v = self.v2(v)
        return p, v

    def get_value(self, x):
        v = F.relu(self.v1(x))
        return self.v2(v)