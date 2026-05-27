# src/train.py
#
# The training loop — this is where the agent actually learns.
# Run this file to start training: python src/train.py
#
# What happens here:
#   - Agent plays through the stock market data 500 times (episodes)
#   - Each episode it gets better at deciding when to buy/sell/hold
#   - We track performance on validation data to know when to stop early

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm   # shows a nice progress bar
import torch

from src.environment import TradingEnvironment
from src.agent import DQNAgent
from config import (
    NUM_EPISODES, TARGET_UPDATE, EARLY_STOP_PATIENCE,
    MODELS_DIR, LOGS_DIR, SEED
)

# Make sure output folders exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)


def run_episode(env, agent, training=True):
    """
    Run one full episode (one pass through the data).

    If training=True  : agent learns and updates weights
    If training=False : agent just acts (used for validation)

    Returns the total reward and final net worth for this episode.
    """
    state = env.reset()
    total_reward = 0
    loss_list    = []

    while True:
        # Agent picks an action
        action = agent.select_action(state)

        # Environment responds
        next_state, reward, done = env.step(action)

        # Store experience (only during training)
        if training:
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                loss_list.append(loss)

        total_reward += reward
        state = next_state

        if done:
            break

    avg_loss = np.mean(loss_list) if loss_list else 0.0
    return total_reward, env.net_worth, avg_loss


def train():
    # Create environments for training and validation
    train_env = TradingEnvironment(split="train")
    val_env   = TradingEnvironment(split="val")

    agent = DQNAgent()

    # Tracking variables
    best_val_worth   = 0           # best net worth seen on validation data
    patience_counter = 0           # how many episodes since last improvement
    history          = []          # stores stats for every episode

    print("Starting training...\n")

    for episode in tqdm(range(1, NUM_EPISODES + 1)):

        # --- Training episode ---
        train_reward, train_worth, avg_loss = run_episode(
            train_env, agent, training=True
        )

        # --- Decay epsilon after each episode ---
        agent.decay_epsilon()

        # --- Update target network every TARGET_UPDATE episodes ---
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        # --- Validation episode (no learning, just measure performance) ---
        # We turn off epsilon so the agent always uses its best action
        saved_epsilon    = agent.epsilon
        agent.epsilon    = 0.0   # no random actions during validation
        val_reward, val_worth, _ = run_episode(val_env, agent, training=False)
        agent.epsilon    = saved_epsilon   # restore epsilon

        # --- Save stats for this episode ---
        history.append({
            "episode":     episode,
            "train_worth": train_worth,
            "val_worth":   val_worth,
            "train_reward":train_reward,
            "val_reward":  val_reward,
            "epsilon":     agent.epsilon,
            "loss":        avg_loss
        })

        # --- Print progress every 50 episodes ---
        if episode % 50 == 0:
            print(f"\nEpisode {episode:4d} | "
                  f"Train: ${train_worth:,.2f} | "
                  f"Val: ${val_worth:,.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.6f}")

        # --- Early stopping ---
        # If validation net worth improves, save the model and reset patience.
        # If it doesn't improve for EARLY_STOP_PATIENCE episodes, stop training.
        if val_worth > best_val_worth:
            best_val_worth   = val_worth
            patience_counter = 0

            # Save the best model so far
            torch.save(agent.online_net.state_dict(),
                       MODELS_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at episode {episode}.")
                print(f"No improvement for {EARLY_STOP_PATIENCE} episodes.")
                break

    print(f"\nTraining complete. Best val net worth: ${best_val_worth:,.2f}")
    print(f"Model saved to: {MODELS_DIR / 'best_model.pt'}")

    # Save history as numpy file for plotting later
    np.save(LOGS_DIR / "history.npy", history)
    print(f"History saved to: {LOGS_DIR / 'history.npy'}")

    return agent, history


# Run training when this file is executed directly
if __name__ == "__main__":
    agent, history = train()