# config.py  — central place for all hyperparameters and paths
from pathlib import Path

ROOT = Path(__file__).parent

# --- Paths ---
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models" / "checkpoints"
RESULTS_DIR     = ROOT / "results"
FIGURES_DIR     = ROOT / "results" / "figures"
LOGS_DIR        = ROOT / "results" / "logs"

# --- Dataset ---
TICKER          = "AAPL"
START_DATE      = "2015-01-01"
END_DATE        = "2023-12-31"
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# test = remaining 0.15

# --- Environment ---
WINDOW_SIZE     = 20        # observation window (timesteps)
INITIAL_BALANCE = 10_000.0
TRANSACTION_FEE = 0.001     # 0.1% per trade

# --- Agent (DQN) ---
STATE_DIM       = WINDOW_SIZE * 5 + 2   # OHLCV * window + balance + position
ACTION_DIM      = 3                      # 0=Hold, 1=Buy, 2=Sell
HIDDEN_DIM      = 128
LEARNING_RATE   = 1e-4
GAMMA           = 0.99                   # discount factor
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.995
BATCH_SIZE      = 64
REPLAY_BUFFER   = 10_000
TARGET_UPDATE   = 10                     # update target net every N episodes

# --- Training ---
NUM_EPISODES    = 500
EARLY_STOP_PATIENCE = 80                 # episodes without val improvement
SEED            = 42