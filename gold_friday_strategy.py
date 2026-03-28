"""
Gold Friday Strategy Test
Tests whether gold (GLD ETF) tends to rise on Fridays.
"""

import pandas as pd
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# --- Download & prepare data ---
df = yf.download("GLD", start="2005-01-01", progress=False)[["Close"]]
df.columns = ["Close"]
df["Return"] = df["Close"].pct_change()
df["Day"] = df.index.dayofweek  # 0=Mon, 4=Fri
df.dropna(inplace=True)

# --- Day-of-week stats ---
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
rows = []
for i, name in enumerate(day_names):
    r = df[df["Day"] == i]["Return"]
    t, p = stats.ttest_1samp(r, 0)
    rows.append({
        "Day": name,
        "Mean Return (%)": round(r.mean() * 100, 4),
        "Win Rate (%)": round((r > 0).mean() * 100, 2),
        "P-Value": round(p, 4),
        "Significant": "Yes" if p < 0.05 else "No",
    })

print(pd.DataFrame(rows).set_index("Day").to_string())

# --- Friday summary ---
fri = df[df["Day"] == 4]["Return"]
print(f"\nFriday: mean={fri.mean()*100:.4f}%, win rate={(fri>0).mean()*100:.1f}%, p={stats.ttest_1samp(fri,0).pvalue:.4f}")

# --- Backtest: hold only on Fridays vs buy-and-hold ---
n_years = len(df) / 252
fri_growth = (1 + fri).cumprod().iloc[-1]
bah_growth = df["Close"].iloc[-1] / df["Close"].iloc[0]

print(f"\nFriday-only CAGR: {(fri_growth**(1/n_years)-1)*100:.2f}%")
print(f"Buy & Hold  CAGR: {(bah_growth**(1/n_years)-1)*100:.2f}%")
