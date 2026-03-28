"""
Gold Friday Strategy Test
Tests whether gold prices tend to rise on Fridays.
Uses GLD ETF as a proxy for gold prices via yfinance.
"""

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


def download_gold_data(ticker="GLD", start="2005-01-01", end=None):
    """Download gold price data using yfinance."""
    print(f"Downloading {ticker} data from {start}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Close"]].copy()
    df.columns = ["Close"]
    df.dropna(inplace=True)
    print(f"Downloaded {len(df)} trading days.")
    return df


def compute_daily_returns(df):
    """Add daily returns and day-of-week columns."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["DayOfWeek"] = df.index.dayofweek  # 0=Monday, 4=Friday
    df["DayName"] = df.index.day_name()
    df.dropna(inplace=True)
    return df


def friday_strategy_stats(df):
    """
    Compute statistics for each day of the week.
    Returns a summary DataFrame.
    """
    summary = []
    for day in range(5):  # Monday=0 ... Friday=4
        day_returns = df[df["DayOfWeek"] == day]["Return"]
        name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][day]
        t_stat, p_value = stats.ttest_1samp(day_returns, 0)
        summary.append({
            "Day": name,
            "Count": len(day_returns),
            "Mean Return (%)": round(day_returns.mean() * 100, 4),
            "Std Dev (%)": round(day_returns.std() * 100, 4),
            "Positive Days (%)": round((day_returns > 0).mean() * 100, 2),
            "T-Statistic": round(t_stat, 4),
            "P-Value": round(p_value, 4),
            "Significant (5%)": "Yes" if p_value < 0.05 else "No",
        })
    return pd.DataFrame(summary).set_index("Day")


def backtest_friday_strategy(df, initial_capital=10_000):
    """
    Backtest: Buy gold at Thursday close, sell at Friday close.
    Compare against buy-and-hold.
    """
    friday_returns = df[df["DayOfWeek"] == 4]["Return"].copy()

    # Strategy: compound returns only on Fridays
    strategy_growth = (1 + friday_returns).cumprod()
    strategy_final = initial_capital * strategy_growth.iloc[-1]

    # Buy and hold over same period
    bah_return = df["Close"].iloc[-1] / df["Close"].iloc[0]
    bah_final = initial_capital * bah_return

    # Annualized return for strategy (252 trading days, ~52 Fridays/year)
    n_years = len(df) / 252
    strategy_cagr = (strategy_growth.iloc[-1] ** (1 / n_years) - 1) * 100
    bah_cagr = (bah_return ** (1 / n_years) - 1) * 100

    print("\n--- Backtest Results ---")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"\nFriday-Only Strategy:")
    print(f"  Final Value:  ${strategy_final:,.2f}")
    print(f"  Total Return: {(strategy_growth.iloc[-1] - 1) * 100:.2f}%")
    print(f"  CAGR:         {strategy_cagr:.2f}%")
    print(f"\nBuy & Hold:")
    print(f"  Final Value:  ${bah_final:,.2f}")
    print(f"  Total Return: {(bah_return - 1) * 100:.2f}%")
    print(f"  CAGR:         {bah_cagr:.2f}%")

    return friday_returns, strategy_growth


def rolling_friday_winrate(df, window=52):
    """Compute rolling win rate for Fridays (rolling 52-week window)."""
    friday_returns = df[df["DayOfWeek"] == 4]["Return"].copy()
    rolling_winrate = friday_returns.gt(0).rolling(window).mean() * 100
    return rolling_winrate


if __name__ == "__main__":
    # --- Download data ---
    df = download_gold_data(ticker="GLD", start="2005-01-01")
    df = compute_daily_returns(df)

    # --- Day-of-week statistics ---
    print("\n--- Day-of-Week Return Statistics ---")
    stats_df = friday_strategy_stats(df)
    print(stats_df.to_string())

    # --- Highlight Friday ---
    friday_stats = stats_df.loc["Friday"]
    print(f"\nFriday mean return: {friday_stats['Mean Return (%)']:.4f}%")
    print(f"Friday positive days: {friday_stats['Positive Days (%)']}%")
    print(f"T-test p-value: {friday_stats['P-Value']} → Statistically significant: {friday_stats['Significant (5%)']}")

    # --- Backtest ---
    friday_returns, strategy_growth = backtest_friday_strategy(df)

    # --- Rolling win rate ---
    rolling_wr = rolling_friday_winrate(df)
    print(f"\n--- Rolling 52-Week Friday Win Rate ---")
    print(f"  Average: {rolling_wr.mean():.2f}%")
    print(f"  Min:     {rolling_wr.min():.2f}%")
    print(f"  Max:     {rolling_wr.max():.2f}%")
    print(f"  Current (last 52 Fridays): {rolling_wr.iloc[-1]:.2f}%")

    # --- Optional: plot ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Gold (GLD) Friday Strategy Analysis", fontsize=14)

        # 1. Mean return by day
        stats_df["Mean Return (%)"].plot(kind="bar", ax=axes[0, 0], color=[
            "green" if x > 0 else "red" for x in stats_df["Mean Return (%)"]
        ])
        axes[0, 0].set_title("Mean Daily Return by Day of Week")
        axes[0, 0].set_ylabel("Return (%)")
        axes[0, 0].axhline(0, color="black", linewidth=0.8)
        axes[0, 0].tick_params(axis="x", rotation=0)

        # 2. Positive day % by day
        stats_df["Positive Days (%)"].plot(kind="bar", ax=axes[0, 1], color="steelblue")
        axes[0, 1].set_title("% Positive Days by Day of Week")
        axes[0, 1].set_ylabel("Win Rate (%)")
        axes[0, 1].axhline(50, color="red", linestyle="--", linewidth=0.8, label="50%")
        axes[0, 1].tick_params(axis="x", rotation=0)
        axes[0, 1].legend()

        # 3. Strategy cumulative growth
        strategy_growth.plot(ax=axes[1, 0], color="gold")
        axes[1, 0].set_title("Friday-Only Strategy Cumulative Growth ($1 → $X)")
        axes[1, 0].set_ylabel("Growth Factor")

        # 4. Rolling 52-week Friday win rate
        rolling_wr.plot(ax=axes[1, 1], color="darkorange")
        axes[1, 1].axhline(50, color="red", linestyle="--", linewidth=0.8, label="50%")
        axes[1, 1].set_title("Rolling 52-Week Friday Win Rate")
        axes[1, 1].set_ylabel("Win Rate (%)")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("gold_friday_strategy.png", dpi=150)
        print("\nPlot saved to gold_friday_strategy.png")
        plt.show()

    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
