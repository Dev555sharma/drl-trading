# src/environment.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config import (
    DATA_RAW, TICKER, WINDOW_SIZE,
    INITIAL_BALANCE, TRANSACTION_FEE,
    TRAIN_RATIO, VAL_RATIO
)


class TradingEnvironment:

    HOLD = 0
    BUY  = 1
    SELL = 2

    MAX_SHARES = 3   # agent can hold at most 3 shares at once

    def __init__(self, split="train"):
        self.split  = split
        raw_df      = self._load_data()
        self.prices = self._split_data(raw_df)
        self.max_steps = len(self.prices) - WINDOW_SIZE - 1
        self.reset()

    def _load_data(self):
        path = DATA_RAW / f"{TICKER}.csv"
        df   = pd.read_csv(path, index_col=0, header=[0, 1])
        df.columns = [col[0] for col in df.columns]
        df   = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # Save raw close prices BEFORE normalization (needed for buy_price tracking)
        self.raw_close = df["Close"].values

        # Normalize each column to 0-1
        df = (df - df.min()) / (df.max() - df.min())
        return df.values

    def _split_data(self, data):
        n         = len(data)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        # Also split raw_close the same way
        if self.split == "train":
            self.raw_close = self.raw_close[:train_end]
            return data[:train_end]
        elif self.split == "val":
            self.raw_close = self.raw_close[train_end:val_end]
            return data[train_end:val_end]
        else:
            self.raw_close = self.raw_close[val_end:]
            return data[val_end:]

    def _get_state(self):
        window        = self.prices[self.current_step : self.current_step + WINDOW_SIZE]
        price_features = window.flatten()

        # Normalized portfolio info
        balance_norm  = self.balance / INITIAL_BALANCE
        shares_norm   = self.shares_held / self.MAX_SHARES

        return np.concatenate([price_features, [balance_norm, shares_norm]])

    def reset(self):
        self.current_step  = 0
        self.balance       = INITIAL_BALANCE
        self.shares_held   = 0
        self.net_worth     = INITIAL_BALANCE
        self.buy_price     = 0.0   # average price paid for current shares
        return self._get_state()

    def step(self, action):
        # Use raw (unnormalized) close price for reward calculation
        # This makes profit/loss meaningful in dollar terms
        current_price = self.raw_close[self.current_step + WINDOW_SIZE]
        reward        = 0.0

        # --- Execute action ---
        if action == self.BUY:
            if self.shares_held < self.MAX_SHARES and self.balance >= current_price:
                cost              = current_price * (1 + TRANSACTION_FEE)
                self.balance     -= cost
                # Track average buy price across all shares held
                self.buy_price    = (
                    (self.buy_price * self.shares_held + current_price)
                    / (self.shares_held + 1)
                )
                self.shares_held += 1
                reward            = 0.0   # neutral on buy

        elif action == self.SELL:
            if self.shares_held > 0:
                revenue           = current_price * (1 - TRANSACTION_FEE)
                profit            = revenue - self.buy_price
                self.balance     += revenue
                self.shares_held -= 1

                # Reward = % profit on this trade (this is the key signal)
                reward = (profit / self.buy_price) * 10

                # Reset buy price if no shares left
                if self.shares_held == 0:
                    self.buy_price = 0.0

        elif action == self.HOLD:
            if self.shares_held > 0:
                # While holding, give small reward if price is above buy price
                # and small penalty if below — encourages timely selling
                unrealized_pct = (current_price - self.buy_price) / self.buy_price
                reward = unrealized_pct * 0.1
            else:
                # Penalize doing nothing when not invested
                reward = -0.001

        # --- Update net worth ---
        self.net_worth  = self.balance + self.shares_held * current_price

        # --- Advance time ---
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done