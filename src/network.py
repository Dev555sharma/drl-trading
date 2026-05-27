# src/network.py
#
# This is the neural network — the "brain" of our agent.
# It takes the market state and outputs a score for each action.
# The agent picks the action with the highest score.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from config import STATE_DIM, ACTION_DIM, HIDDEN_DIM


class DQNetwork(nn.Module):
    """
    A simple 3-layer neural network.

    Input  : state vector (102 numbers)
    Output : 3 Q-values, one per action (Hold, Buy, Sell)

    Q-value means: "estimated total future reward if I take this action now"
    Higher Q-value = better action in this situation.
    """

    def __init__(self):
        super(DQNetwork, self).__init__()

        # Three layers: input -> hidden -> hidden -> output
        # ReLU activation just means: "if the value is negative, make it 0"
        # This helps the network learn non-linear patterns
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),  # 102 -> 128
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), # 128 -> 128
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM)  # 128 -> 3
        )

    def forward(self, x):
        """
        Forward pass: feed the state through the network and get Q-values back.
        Called automatically when you do: network(state)
        """
        return self.network(x)