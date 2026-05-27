# src/replay_buffer.py
#
# The replay buffer is the agent's memory.
# It stores past experiences and gives back random batches for learning.
# This is what makes DQN stable — without it, training diverges.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import random
from collections import deque
from config import REPLAY_BUFFER


class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.
    When full, automatically removes the oldest experience (deque does this).
    """

    def __init__(self):
        # deque with maxlen = our buffer size (e.g. 10,000)
        # When it's full and you add a new item, the oldest is automatically removed
        self.buffer = deque(maxlen=REPLAY_BUFFER)

    def add(self, state, action, reward, next_state, done):
        """Save one experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly pick 'batch_size' experiences from memory.
        Returns them as separate numpy arrays — one per component.
        """
        batch = random.sample(self.buffer, batch_size)

        # Unzip the list of tuples into 5 separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32)
        )

    def __len__(self):
        """Lets us do: len(buffer) to check how many experiences are stored."""
        return len(self.buffer)