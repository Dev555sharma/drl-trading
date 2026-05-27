# src/agent.py
#
# The DQN Agent — this is the decision-maker.
# It uses two networks (online + target), a replay buffer, and epsilon-greedy
# exploration to learn which actions lead to the most profit.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import DQNetwork
from src.replay_buffer import ReplayBuffer
from config import (
    ACTION_DIM, LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    BATCH_SIZE, TARGET_UPDATE, SEED
)

# Set random seeds so results are reproducible
torch.manual_seed(SEED)
np.random.seed(SEED)


class DQNAgent:

    def __init__(self):
        # --- Two networks ---
        # online_net  : learns every step (we update this one)
        # target_net  : frozen copy, updated every TARGET_UPDATE episodes
        # Why two? Because if we update the target every step, the agent
        # chases a moving bullseye and training becomes unstable.
        self.online_net = DQNetwork()
        self.target_net = DQNetwork()

        # Copy online_net weights into target_net at the start
        self.target_net.load_state_dict(self.online_net.state_dict())

        # target_net never needs gradients — it's just for reference
        self.target_net.eval()

        # Adam optimizer — updates the online_net weights during learning
        self.optimizer = optim.Adam(self.online_net.parameters(),
                                    lr=LEARNING_RATE)

        # Memory
        self.memory = ReplayBuffer()

        # Epsilon: starts at 1.0 (fully random), decays toward 0.01
        self.epsilon = EPSILON_START

        # Count training steps (used to decide when to update target_net)
        self.episode_count = 0

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon  → pick a RANDOM action (explore)
        With probability 1-epsilon → pick the BEST action from the network (exploit)

        Early in training epsilon is high (lots of exploration).
        Over time it decays so the agent trusts its learned knowledge more.
        """
        if np.random.random() < self.epsilon:
            # Random action: 0, 1, or 2
            return np.random.randint(ACTION_DIM)
        else:
            # Use the network: convert state to tensor, get Q-values, pick best
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # add batch dim
            with torch.no_grad():  # no gradient needed for just selecting action
                q_values = self.online_net(state_tensor)
            return q_values.argmax().item()  # index of highest Q-value

    def store_experience(self, state, action, reward, next_state, done):
        """Save one experience to the replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a batch from memory and update the network.
        This is the core of DQN — the Bellman equation in code.

        We only learn once the buffer has enough experiences (>= BATCH_SIZE).
        """
        if len(self.memory) < BATCH_SIZE:
            return None  # not enough memories yet, skip

        # --- Sample a random batch from memory ---
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert everything to PyTorch tensors
        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        # --- Compute current Q-values ---
        # For each experience, get the Q-value of the action that was taken
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Compute target Q-values (Bellman equation) ---
        # Target = reward + gamma * max(Q(next_state))   if not done
        # Target = reward                                 if done (episode ended)
        #
        # This is THE key equation of Q-learning:
        # "The value of this action = immediate reward + discounted best future value"
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        # --- Compute loss and update ---
        # Loss = how far off our Q-value predictions are from the targets
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()  # clear old gradients
        loss.backward()             # compute new gradients
        self.optimizer.step()       # update network weights

        return loss.item()

    def decay_epsilon(self):
        """
        After each episode, reduce epsilon so the agent explores less
        and relies on its learned policy more.
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        """
        Copy online_net weights into target_net.
        Called every TARGET_UPDATE episodes.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())