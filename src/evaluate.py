# src/evaluate.py
#
# Loads the best saved model and evaluates it on the TEST set.
# The test set is data the agent has NEVER seen — this is the true score.
#
# Produces:
#   1. Performance metrics (Sharpe ratio, max drawdown, total return)
#   2. An equity curve plot saved to results/figures/

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.environment import TradingEnvironment
from src.agent import DQNAgent
from config import MODELS_DIR, FIGURES_DIR, INITIAL_BALANCE

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_best_agent():
    """Load the agent and restore the best saved weights."""
    agent = DQNAgent()
    weights = torch.load(MODELS_DIR / "best_model.pt")
    agent.online_net.load_state_dict(weights)
    agent.epsilon = 0.0   # no random actions — use learned policy only
    return agent


def run_test_episode(env, agent):
    """
    Run one episode on the test environment.
    Records net worth at every step so we can plot the equity curve.
    """
    state = env.reset()
    net_worth_history = [INITIAL_BALANCE]
    action_history    = []

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        net_worth_history.append(env.net_worth)
        action_history.append(action)

        state = next_state
        if done:
            break

    return np.array(net_worth_history), np.array(action_history)


def compute_metrics(net_worth_history):
    """
    Compute the 3 key metrics used to judge any trading strategy.

    1. Total return     : how much money did we make overall (%)
    2. Sharpe ratio     : return adjusted for risk (higher = better)
                          > 1.0 is good, > 2.0 is excellent
    3. Max drawdown     : worst peak-to-trough drop (lower = better)
                          tells you the worst losing streak
    """
    returns = np.diff(net_worth_history) / net_worth_history[:-1]

    total_return = (net_worth_history[-1] - net_worth_history[0]) / net_worth_history[0]
    total_return_pct = total_return * 100

    # Sharpe ratio: mean daily return / std of daily returns * sqrt(252)
    # 252 = trading days in a year (annualizes the ratio)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown: biggest % drop from any peak to any following trough
    peak = net_worth_history[0]
    max_drawdown = 0.0
    for worth in net_worth_history:
        if worth > peak:
            peak = worth
        drawdown = (peak - worth) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    max_drawdown_pct = max_drawdown * 100

    return {
        "Final net worth":   f"${net_worth_history[-1]:,.2f}",
        "Total return":      f"{total_return_pct:+.2f}%",
        "Sharpe ratio":      f"{sharpe:.3f}",
        "Max drawdown":      f"{max_drawdown_pct:.2f}%",
    }


def plot_equity_curve(net_worth_history, buy_hold_history):
    """
    Plot our agent's net worth over time vs a simple buy-and-hold strategy.
    This is the main result figure for your GitHub.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(net_worth_history, label="DQN Agent",      color="#2563eb", linewidth=2)
    ax.plot(buy_hold_history,  label="Buy & Hold",     color="#9ca3af",
            linewidth=1.5, linestyle="--")

    ax.axhline(y=INITIAL_BALANCE, color="#ef4444", linewidth=1,
               linestyle=":", label="Starting capital")

    ax.set_title("DQN Agent vs Buy & Hold — Test Set", fontsize=14)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / "equity_curve.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nFigure saved to: {save_path}")


def compute_buy_hold(env):
    """
    Baseline strategy: buy on day 1, hold until the end.
    We compare our agent against this — if we can't beat buy-and-hold,
    the agent hasn't learned anything useful.
    """
    # First close price in test data
    start_price = env.prices[0][3]
    # How many shares could we buy with our initial balance?
    shares = INITIAL_BALANCE // start_price if start_price > 0 else 0
    remaining_cash = INITIAL_BALANCE - shares * start_price

    history = []
    for i in range(len(env.prices)):
        current_price = env.prices[i][3]
        worth = remaining_cash + shares * current_price
        history.append(worth)

    return np.array(history)


def evaluate():
    print("Loading best model...")
    agent = load_best_agent()

    print("Running on test set...")
    test_env = TradingEnvironment(split="test")

    net_worth_history, action_history = run_test_episode(test_env, agent)
    buy_hold_history  = compute_buy_hold(test_env)

    # Print metrics
    print("\n" + "="*40)
    print("TEST SET RESULTS")
    print("="*40)
    metrics = compute_metrics(net_worth_history)
    for key, val in metrics.items():
        print(f"  {key:<20}: {val}")

    # Action breakdown
    actions = ["Hold", "Buy", "Sell"]
    print("\nAction breakdown:")
    for i, name in enumerate(actions):
        count = (action_history == i).sum()
        pct   = count / len(action_history) * 100
        print(f"  {name:<6}: {count:4d} times ({pct:.1f}%)")

    # Plot
    plot_equity_curve(net_worth_history, buy_hold_history)


if __name__ == "__main__":
    evaluate()