import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid")
SIDE_COLORS = {"BUY": "darkgray", "SELL": "lightgray"}

df = pd.read_csv("trader_analysis.csv")

df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_", regex=False))


rename_map = {
    "coin": "symbol",
    "execution_price": "exec_price",
    "timestamp_ist": "ts_ist",
    "start_position": "start_pos",
}
df.rename(columns=rename_map, inplace=True)

df["ts_ist"] = pd.to_datetime(df["ts_ist"], format="%d-%m-%Y %H:%M", errors="coerce")
ts_null_mask = df["ts_ist"].isna()
if ts_null_mask.any():
    df.loc[ts_null_mask, "ts_ist"] = pd.to_datetime(
        df.loc[ts_null_mask, "timestamp"], unit="ms", errors="coerce"
    )
df.dropna(subset=["ts_ist"], inplace=True)

for col in ["exec_price", "size_tokens", "size_usd", "closed_pnl", "fee", "start_pos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["side"] = df["side"].str.strip().str.upper()
df["direction"] = df["direction"].str.strip().str.title()
df["symbol"] = df["symbol"].str.strip().str.upper()

df.drop_duplicates(subset=["trade_id"], keep="last", inplace=True)
df.dropna(subset=["exec_price", "size_usd", "closed_pnl"], inplace=True)
df.sort_values("ts_ist", inplace=True)
df.reset_index(drop=True, inplace=True)
df["date"] = df["ts_ist"].dt.date
df["hour"] = df["ts_ist"].dt.hour
df["net_pnl"] = df["closed_pnl"] - df["fee"].fillna(0)

df["size_bucket"] = pd.cut(
    df["size_usd"],
    bins=[0, 100, 1_000, 10_000, 100_000, np.inf],
    labels=["<$100", "$100-1K", "$1K-10K", "$10K-100K", ">$100K"],
    right=True
)

closed = df[df["closed_pnl"] != 0].copy()
wins = closed[closed["closed_pnl"] > 0]["closed_pnl"]
losses = closed[closed["closed_pnl"] < 0]["closed_pnl"]

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Closed PnL Distribution", fontsize=13)

axes1[0].hist(wins.clip(upper=wins.quantile(0.99)), bins=60, color="darkgray", edgecolor="black")
axes1[0].set_title("Winning Trades (Closed PnL > 0)")
axes1[0].set_xlabel("PnL (USD)")
axes1[0].set_ylabel("Frequency")
axes1[0].axvline(wins.mean(), color="black", linestyle="--", label=f"Mean=${wins.mean():.1f}")
axes1[0].legend()

axes1[1].hist(losses.clip(lower=losses.quantile(0.01)), bins=60, color="darkgray", edgecolor="black")
axes1[1].set_title("Losing Trades (Closed PnL < 0)")
axes1[1].set_xlabel("PnL (USD)")
axes1[1].set_ylabel("Frequency")
axes1[1].axvline(losses.mean(), color="black", linestyle="--", label=f"Mean=${losses.mean():.1f}")
axes1[1].legend()

plt.tight_layout()
plt.show()

top_n_traders = 10
trader_pnl = df.groupby("account")["closed_pnl"].sum().sort_values(ascending=False).head(top_n_traders)
short_labels = [a[:8] + "…" for a in trader_pnl.index]

fig2, ax2 = plt.subplots(figsize=(12, 5))
bars = ax2.bar(short_labels, trader_pnl.values, color="darkgray", edgecolor="black")
ax2.set_title(f"Top {top_n_traders} Traders by Total Closed PnL", fontsize=13)
ax2.set_xlabel("Account (truncated)")
ax2.set_ylabel("Total PnL (USD)")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

for bar, val in zip(bars, trader_pnl.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(trader_pnl.values.max()) * 0.01,
            f"${val:,.0f}", ha="center", fontsize=8, color="black")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

top_n_symbols = 15
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("Symbol-level Analysis", fontsize=13)

sym_cnt = df["symbol"].value_counts().head(top_n_symbols)
axes3[0].barh(sym_cnt.index[::-1], sym_cnt.values[::-1], color="darkgray", edgecolor="black")
axes3[0].set_title(f"Top {top_n_symbols} Symbols by Trade Count")
axes3[0].set_xlabel("Number of Trades")

sym_pnl = df.groupby("symbol")["closed_pnl"].sum().sort_values(ascending=False).head(top_n_symbols)
axes3[1].barh(sym_pnl.index[::-1], sym_pnl.values[::-1], color="lightgray", edgecolor="black")
axes3[1].set_title(f"Top {top_n_symbols} Symbols by Total PnL")
axes3[1].set_xlabel("Total Closed PnL (USD)")
axes3[1].axvline(0, color="black", linewidth=0.8)
axes3[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.show()

fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle("Buy vs Sell Analysis", fontsize=13)

side_cnt = df["side"].value_counts()
axes4[0].pie(side_cnt.values, labels=side_cnt.index, 
            colors=[SIDE_COLORS.get(s, "gray") for s in side_cnt.index],
            autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "black"})
axes4[0].set_title("Trade Count")

side_vol = df.groupby("side")["size_usd"].sum()
axes4[1].bar(side_vol.index, side_vol.values, 
            color=[SIDE_COLORS.get(s, "gray") for s in side_vol.index], edgecolor="black")
axes4[1].set_title("Total Volume (USD)")
axes4[1].set_ylabel("USD")
axes4[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

side_pnl = df.groupby("side")["closed_pnl"].mean()
bars_side = axes4[2].bar(side_pnl.index, side_pnl.values, 
                       color=[SIDE_COLORS.get(s, "gray") for s in side_pnl.index], edgecolor="black")
axes4[2].set_title("Average Closed PnL")
axes4[2].set_ylabel("Avg PnL (USD)")
axes4[2].axhline(0, color="black", linewidth=0.8)
for bar, val in zip(bars_side, side_pnl.values):
    axes4[2].text(bar.get_x() + bar.get_width()/2, val + 0.3, f"${val:.2f}", 
                 ha="center", fontsize=9, color="black")

plt.tight_layout()
plt.show()

hourly_cnt = df.groupby("hour").size()
hourly_pnl = df.groupby("hour")["closed_pnl"].mean()
fig5, axes5 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig5.suptitle("Intraday Trading Patterns (IST)", fontsize=13)
axes5[0].bar(hourly_cnt.index, hourly_cnt.values, color="darkgray", edgecolor="black")
axes5[0].set_ylabel("Number of Trades")
axes5[0].set_title("Trade Activity by Hour")
axes5[1].bar(hourly_pnl.index, hourly_pnl.values, color="lightgray", edgecolor="black")
axes5[1].axhline(0, color="black", linewidth=0.8)
axes5[1].set_ylabel("Avg Closed PnL (USD)")
axes5[1].set_xlabel("Hour of Day (IST)")
axes5[1].set_title("Average PnL by Hour")
axes5[1].set_xticks(range(0, 24))
plt.tight_layout()
plt.show()

top_n_cum = 5
top_traders_cum = df.groupby("account")["closed_pnl"].sum().nlargest(top_n_cum).index.tolist()
fig6, ax6 = plt.subplots(figsize=(16, 6))
linestyles = ['-', '--', '-.', ':', '-']

for i, acc in enumerate(top_traders_cum):
    sub = df[df["account"] == acc].sort_values("ts_ist").copy()
    sub["cum_pnl"] = sub["closed_pnl"].cumsum()
    ax6.plot(sub["ts_ist"], sub["cum_pnl"], linewidth=1.5, 
            label=acc[:10] + "…", color="black", linestyle=linestyles[i%len(linestyles)])

ax6.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax6.set_title(f"Cumulative PnL — Top {top_n_cum} Traders", fontsize=13)
ax6.set_xlabel("Date")
ax6.set_ylabel("Cumulative PnL (USD)")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax6.legend(fontsize=9)

plt.tight_layout()
plt.show()

q99_pnl = closed["closed_pnl"].quantile(0.99)
q01_pnl = closed["closed_pnl"].quantile(0.01)
plot_df = closed[(closed["closed_pnl"] <= q99_pnl) & (closed["closed_pnl"] >= q01_pnl)]

fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.scatter(plot_df["fee"], plot_df["closed_pnl"], 
           c="darkgray", edgecolor="black", s=15, alpha=0.6)

ax7.axhline(0, color="black", linewidth=1, linestyle="--")
ax7.set_title("Fee vs Closed PnL (closed trades only)", fontsize=13)
ax7.set_xlabel("Fee Paid (USD)")
ax7.set_ylabel("Closed PnL (USD)")

plt.tight_layout()
plt.show()

closed_sizes = df[(df["closed_pnl"] != 0) & df["size_bucket"].notna()].copy()
order = ["<$100", "$100-1K", "$1K-10K", "$10K-100K", ">$100K"]

fig8, ax8 = plt.subplots(figsize=(12, 6))
sns.boxplot(data=closed_sizes, x="size_bucket", y="closed_pnl", 
            order=order, color="lightgray", ax=ax8, showfliers=False)
ax8.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax8.set_title("PnL Distribution by Trade Size Bucket", fontsize=13)
ax8.set_xlabel("Trade Size (USD)")
ax8.set_ylabel("Closed PnL (USD)")   
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

closed_winrate = df[df["closed_pnl"] != 0].copy()
closed_winrate["win"] = (closed_winrate["closed_pnl"] > 0).astype(int)
stats = (closed_winrate.groupby("account")
               .agg(trades=("win", "count"), win_rate=("win", "mean"))
               .query("trades >= 20")
               .sort_values("win_rate", ascending=False))
stats["win_rate_pct"] = stats["win_rate"] * 100
short_labels_win = [a[:8] + "…" for a in stats.index]
fig9, ax9 = plt.subplots(figsize=(12, 5))
ax9.bar(short_labels_win, stats["win_rate_pct"].values, color="darkgray", edgecolor="black")
ax9.axhline(50, color="black", linestyle="--", linewidth=1.2, label="50% break-even")
ax9.set_title("Win Rate by Trader Account (min 20 closed trades)", fontsize=13)
ax9.set_xlabel("Account (truncated)")
ax9.set_ylabel("Win Rate (%)")
ax9.legend()
for i, (v, t) in enumerate(zip(stats["win_rate_pct"], stats["trades"])):
    ax9.text(i, v + 0.5, f"{v:.1f}%\n(n={t})", ha="center", fontsize=8, color="black")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

df_monthly = df.copy()
df_monthly["ym"] = df_monthly["ts_ist"].dt.to_period("M")
monthly = df_monthly.groupby(["ym", "side"])["size_usd"].sum().unstack(fill_value=0)
fig10, ax10 = plt.subplots(figsize=(16, 5))
monthly.plot(kind="bar", stacked=True, 
             color=[SIDE_COLORS.get(c, "gray") for c in monthly.columns], 
             edgecolor="black", ax=ax10) 
ax10.set_title("Monthly Trade Volume by Side", fontsize=13)
ax10.set_xlabel("Month")
ax10.set_ylabel("Volume (USD)")
ax10.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax10.legend(title="Side")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()