import yfinance as yf
from config import TICKER, START_DATE, END_DATE, DATA_RAW

# Make sure the folder exists
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Download the data from Yahoo Finance
print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Save to CSV
save_path = DATA_RAW / f"{TICKER}.csv"
df.to_csv(save_path)

print(f"Saved {len(df)} rows to {save_path}")
print(df.tail())   # show last 5 rows so you can verify it worked