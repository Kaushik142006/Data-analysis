import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid")

CATEGORY_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# 2. Load and preprocess data
df = pd.read_csv("sentiment_analysis.csv")
df["date"] = pd.to_datetime(df["date"])
df["classification"] = pd.Categorical(df["classification"].str.strip().str.title(), 
                                      categories=CATEGORY_ORDER, ordered=True)
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.day_name()
df["rolling_30d"] = df["value"].rolling(30, min_periods=1).mean()

fig1, ax1 = plt.subplots(figsize=(14, 5))
fig1.suptitle("Bitcoin Fear & Greed Index — Full History", fontsize=14)

sns.scatterplot(data=df, x="date", y="value", hue="classification", 
                hue_order=CATEGORY_ORDER, palette="RdYlGn", s=30, ax=ax1)
ax1.plot(df["date"], df["rolling_30d"], color="black", linewidth=1.5, label="30-day MA")
ax1.set_xlim(df["date"].min(), df["date"].max())
ax1.set_ylim(0, 100)
ax1.set_xlabel("Date"); ax1.set_ylabel("Fear & Greed Value")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()


fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Sentiment Distribution", fontsize=13)
counts = df["classification"].value_counts().reindex(CATEGORY_ORDER)
sns.barplot(x=counts.index, y=counts.values, ax=axes2[0], palette="RdYlGn")
axes2[0].set_title("Frequency of Each Sentiment Category")
axes2[0].set_xlabel("Sentiment"); axes2[0].set_ylabel("Number of Days")
axes2[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(counts.values):
    axes2[0].text(i, v + 0.5, str(v), ha="center", fontsize=9)
axes2[1].hist(df["value"], bins=30, edgecolor="black", alpha=0.7)
axes2[1].axvline(df["value"].mean(), color="red", linestyle="--", label=f"Mean={df['value'].mean():.1f}")
axes2[1].axvline(df["value"].median(), color="green", linestyle=":", label=f"Median={df['value'].median():.1f}")
axes2[1].set_title("Distribution of Fear & Greed Values")
axes2[1].set_xlabel("Value (0=Extreme Fear, 100=Extreme Greed)")
axes2[1].set_ylabel("Frequency")
axes2[1].legend()
plt.tight_layout()
plt.show()
pivot = df.groupby(["year", "month"])["value"].mean().unstack(level="month")
pivot.columns = [pd.to_datetime(str(m), format='%m').strftime('%b') for m in pivot.columns]
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100, linewidths=0.5, ax=ax3, cbar_kws={"label": "Avg Fear & Greed Value"})
ax3.set_title("Monthly Average Fear & Greed Value by Year", fontsize=13)
ax3.set_xlabel("Month"); ax3.set_ylabel("Year")
plt.tight_layout()
plt.show()
fig4, ax4 = plt.subplots(figsize=(10, 4))
sns.boxplot(data=df, x="year", y="value", palette="RdYlGn", ax=ax4)
ax4.set_title("Year-wise Fear & Greed Value Distribution", fontsize=13)
ax4.set_xlabel("Year"); ax4.set_ylabel("Fear & Greed Value")
ax4.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Neutral (50)")
ax4.legend()
plt.tight_layout()
plt.show()
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow = df.groupby("day_of_week")["value"].mean().reindex(order)
fig5, ax5 = plt.subplots(figsize=(10, 4))
sns.barplot(x=dow.index, y=dow.values, ax=ax5, palette="RdYlGn")
ax5.set_title("Average Fear & Greed Value by Day of Week", fontsize=13)
ax5.set_xlabel("Day of Week"); ax5.set_ylabel("Avg Value")
ax5.axhline(50, color="gray", linestyle="--", linewidth=0.8)
for i, v in enumerate(dow.values):
    ax5.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.show()
df2 = df.copy()
df2["next_class"] = df2["classification"].shift(-1)
df2.dropna(subset=["next_class"], inplace=True)
matrix = pd.crosstab(df2["classification"], df2["next_class"], normalize="index").reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=ax6, cbar_kws={"label": "Transition Probability"})
ax6.set_title("Sentiment Transition Matrix (Day → Next Day)", fontsize=12)
ax6.set_xlabel("Next Day Sentiment"); ax6.set_ylabel("Current Day Sentiment")
plt.tight_layout()
plt.show()
df2["rolling_std"] = df2["value"].rolling(30, min_periods=10).std()
fig7, axes7 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig7.suptitle("Fear & Greed Index & Rolling Volatility (30-day)", fontsize=13)
axes7[0].plot(df2["date"], df2["value"], color="steelblue", linewidth=1.5, label="Index Value")
axes7[0].plot(df2["date"], df2["rolling_30d"], color="black", linewidth=1.5, label="30-day MA")
axes7[0].set_ylabel("F&G Value"); axes7[0].legend()
axes7[1].fill_between(df2["date"], df2["rolling_std"], alpha=0.5, color="gray")
axes7[1].set_ylabel("30-day Std Dev")
axes7[1].set_xlabel("Date")
plt.tight_layout()
plt.show()