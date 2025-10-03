import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
csv_path = BASE / "data" / "raw" / "rh_data.csv"  # nome correto do arquivo

df = pd.read_csv(csv_path)
print(df.shape)
print(df.head())
print(df.info())
