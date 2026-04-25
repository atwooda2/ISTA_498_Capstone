import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "replay_csvs" / "master_replays.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# ── Cleaning — run once, applies to every plot ────────────────────────────────
df = df[df["rank"] != ".ipynb_checkpoints"].reset_index(drop=True)
df["smurf"] = df["smurf"].fillna(0).astype(int)

# Force numeric on stat columns that may have been read as strings
stat_cols = [
    "score", "goals", "assists", "saves", "shots", "avg speed",
    "shooting percentage", "avg boost amount", "amount collected",
    "amount stolen", "bpm",
    "percentage slow speed", "percentage boost speed", "percentage supersonic speed",
    "percentage on ground", "percentage low in air", "percentage high in air",
    "percentage defensive third", "percentage neutral third", "percentage offensive third",
    "time behind ball", "time in front of ball",
]
for col in stat_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Shared config ─────────────────────────────────────────────────────────────
NORMAL_COLOR = "#378ADD"
SMURF_COLOR  = "#dfa53b"

norm  = df[df["smurf"] == 0]
smurf = df[df["smurf"] == 1]

rank_order = [
    "bronze-1", "bronze-2", "bronze-3",
    "silver-1", "silver-2", "silver-3",
    "gold-1", "gold-2", "gold-3",
    "platinum-1", "platinum-2", "platinum-3",
    "diamond-1", "diamond-2", "diamond-3",
    "champion-1", "champion-2", "champion-3",
    "grand-champion-1", "grand-champion-2", "grand-champion-3",
    "supersonic-legend",
]
rank_labels = [
    "B1", "B2", "B3", "S1", "S2", "S3", "G1", "G2", "G3",
    "P1", "P2", "P3", "D1", "D2", "D3", "C1", "C2", "C3",
    "GC1", "GC2", "GC3", "SSL",
]

tier_map = {
    "bronze-1": "Beginner",        "bronze-2": "Beginner",        "bronze-3": "Beginner",
    "silver-1": "Beginner",        "silver-2": "Beginner",        "silver-3": "Beginner",
    "gold-1": "Intermediate",      "gold-2": "Intermediate",      "gold-3": "Intermediate",
    "platinum-1": "Intermediate",  "platinum-2": "Intermediate",  "platinum-3": "Intermediate",
    "diamond-1": "Intermediate",   "diamond-2": "Intermediate",   "diamond-3": "Intermediate",
    "champion-1": "Expert",      "champion-2": "Expert",      "champion-3": "Expert",
    "grand-champion-1": "Expert","grand-champion-2": "Expert","grand-champion-3": "Expert",
    "supersonic-legend": "Expert",
}
df["tier"] = df["rank"].map(tier_map)
df.loc[df["smurf"] == 1, "tier"] = "Smurf"


# ── Plot 1+2 Combined: Avg score by rank, smurf bar shown where smurfs exist ──
norm_score_by_rank  = [norm[norm["rank"] == r]["score"].mean() for r in rank_order]
smurf_score_by_rank = [
    smurf[smurf["rank"] == r]["score"].mean() if len(smurf[smurf["rank"] == r]) > 0 else np.nan
    for r in rank_order
]

x = np.arange(len(rank_order))
w = 0.4

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, norm_score_by_rank,  w, label="Normal", color=NORMAL_COLOR, alpha=0.85, edgecolor="white")
ax.bar(x + w/2, smurf_score_by_rank, w, label="Smurf",  color=SMURF_COLOR,  alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(rank_labels, rotation=45, ha="right")
ax.set_ylabel("Average Score")
ax.set_title("Average Score by Rank: Normal Players vs Smurfs\n(smurf bar only shown where smurfs were detected in that rank)")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 3: Radar chart ───────────────────────────────────────────────────────
stats      = ["goals", "assists", "saves"]
vals_norm  = norm[stats].mean().values
vals_smurf = smurf[stats].mean().values

angles     = np.linspace(0, 2 * np.pi, len(stats), endpoint=False)
vals_norm  = np.concatenate((vals_norm,  [vals_norm[0]]))
vals_smurf = np.concatenate((vals_smurf, [vals_smurf[0]]))
angles     = np.concatenate((angles,     [angles[0]]))

fig = plt.figure(figsize=(6, 6))
ax  = plt.subplot(111, polar=True)
ax.plot(angles, vals_norm,  label="Normal", linewidth=2, color=NORMAL_COLOR)
ax.plot(angles, vals_smurf, label="Smurf",  linewidth=2, color=SMURF_COLOR)
ax.fill(angles, vals_norm,  alpha=0.2, color=NORMAL_COLOR)
ax.fill(angles, vals_smurf, alpha=0.2, color=SMURF_COLOR)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(stats)
plt.title("Radar Chart: Gameplay Stats Comparison")
plt.legend()
plt.show()

# ── Plot 4 & 5: Correlation heatmaps ─────────────────────────────────────────
heatmap_cols = ["score", "goals", "assists", "saves", "shots", "avg speed"]

plt.figure(figsize=(10, 6))
sns.heatmap(smurf[heatmap_cols].corr(), annot=True, cmap="Reds")
plt.title("Smurf Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(norm[heatmap_cols].corr(), annot=True, cmap="Blues")
plt.title("Normal Player Correlation Heatmap")
plt.show()

# ── Plot 6: Speed KDE ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.kdeplot(norm["avg speed"].dropna(),  fill=True, label="Normal", color=NORMAL_COLOR)
sns.kdeplot(smurf["avg speed"].dropna(), fill=True, label="Smurf",  color=SMURF_COLOR)
plt.title("Speed Distribution: Normal vs Smurf")
plt.xlabel("Average Speed")
plt.legend()
plt.show()

# ── Plot 7: Boost boxplot ─────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="smurf", y="avg boost amount", palette="coolwarm")
plt.xticks([0, 1], ["Normal", "Smurf"])
plt.title("Boost Usage Comparison")
plt.show()

# ── Plot 8: Scatter — field positioning by tier ───────────────────────────────
tier_styles = {
    "Beginner":     {"color": "#4CAF50",   "alpha": 0.35, "s": 10},
    "Intermediate": {"color": "#FF9800",   "alpha": 0.35, "s": 10},
    "Expert":       {"color": "#9C27B0",   "alpha": 0.35, "s": 10},
    "Smurf":        {"color": SMURF_COLOR, "alpha": 0.95, "s": 30},
}

fig, ax = plt.subplots(figsize=(10, 6))
for tier, style in tier_styles.items():
    subset = df[df["tier"] == tier]
    ax.scatter(
        subset["time behind ball"],
        subset["time in front of ball"],
        c=style["color"],
        alpha=style["alpha"],
        s=style["s"],
        label=f"{tier} (n={len(subset)})",
    )
ax.set_xlabel("Time Behind Ball")
ax.set_ylabel("Time In Front of Ball")
ax.set_title("Field Positioning: Beginner / Intermediate / Advanced / Smurf")
ax.legend(title="Tier", markerscale=2)
plt.tight_layout()
plt.show()

# ── Plot 9: Score distribution histogram ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 2000]
ax.hist(norm["score"].dropna(),  bins=bins, color=NORMAL_COLOR, alpha=0.6, label="Normal", edgecolor="white")
ax.hist(smurf["score"].dropna(), bins=bins, color=SMURF_COLOR,  alpha=0.8, label="Smurf",  edgecolor="white")
ax.axvline(norm["score"].median(),  color=NORMAL_COLOR, linestyle="--", linewidth=1.5,
           label=f"Normal median ({int(norm['score'].median())})")
ax.axvline(smurf["score"].median(), color=SMURF_COLOR,  linestyle="--", linewidth=1.5,
           label=f"Smurf median ({int(smurf['score'].median())})")
ax.set_xlabel("Score")
ax.set_ylabel("Player count")
ax.set_title("Score Distribution: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 10: Shooting % histogram ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins_pct = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
ax.hist(norm["shooting percentage"].dropna(),  bins=bins_pct, color=NORMAL_COLOR,
        alpha=0.6, label="Normal", edgecolor="white")
ax.hist(smurf["shooting percentage"].dropna(), bins=bins_pct, color=SMURF_COLOR,
        alpha=0.8, label="Smurf",  edgecolor="white")
ax.set_xlabel("Shooting percentage (%)")
ax.set_ylabel("Player count")
ax.set_title("Shooting % Distribution: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 11: Speed profile grouped bar ───────────────────────────────────────
speed_cats  = ["Slow speed", "Boost speed", "Supersonic"]
speed_cols  = ["percentage slow speed", "percentage boost speed", "percentage supersonic speed"]
norm_means  = [norm[c].mean()  for c in speed_cols]
smurf_means = [smurf[c].mean() for c in speed_cols]

x, w = np.arange(len(speed_cats)), 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, norm_means,  w, label="Normal", color=NORMAL_COLOR)
ax.bar(x + w/2, smurf_means, w, label="Smurf",  color=SMURF_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(speed_cats)
ax.set_ylabel("% of game time")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Speed Profile: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 12: Air vs ground grouped bar ───────────────────────────────────────
air_cats  = ["On ground", "Low air", "High air"]
air_cols  = ["percentage on ground", "percentage low in air", "percentage high in air"]
norm_air  = [norm[c].mean()  for c in air_cols]
smurf_air = [smurf[c].mean() for c in air_cols]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, norm_air,  w, label="Normal", color=NORMAL_COLOR)
ax.bar(x + w/2, smurf_air, w, label="Smurf",  color=SMURF_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(air_cats)
ax.set_ylabel("% of game time")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Air vs Ground: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 13: Avg speed by rank — line with smurf reference ───────────────────
rank_speed      = df[df["rank"].isin(rank_order)].groupby("rank")["avg speed"].mean().reindex(rank_order)
smurf_speed_avg = smurf["avg speed"].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rank_labels, rank_speed.values, color=NORMAL_COLOR, marker="o", linewidth=2, label="Normal by rank")
ax.axhline(smurf_speed_avg, color=SMURF_COLOR, linestyle="--", linewidth=2,
           label=f"Smurf avg ({smurf_speed_avg:.0f})")
ax.set_xlabel("Rank")
ax.set_ylabel("Avg speed")
ax.set_title("Avg Speed by Rank vs Smurf Group")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 14: Supersonic % by rank — line with smurf reference ────────────────
rank_sup      = df[df["rank"].isin(rank_order)].groupby("rank")["percentage supersonic speed"].mean().reindex(rank_order)
smurf_sup_avg = smurf["percentage supersonic speed"].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rank_labels, rank_sup.values, color=NORMAL_COLOR, marker="o", linewidth=2, label="Normal by rank")
ax.axhline(smurf_sup_avg, color=SMURF_COLOR, linestyle="--", linewidth=2,
           label=f"Smurf avg ({smurf_sup_avg:.1f}%)")
ax.set_xlabel("Rank")
ax.set_ylabel("% supersonic")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Supersonic % by Rank vs Smurf Group")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 15: Boost management grouped bar ─────────────────────────────────────
boost_cats  = ["Boost collected", "Boost stolen", "Avg boost amount"]
boost_cols  = ["amount collected", "amount stolen", "avg boost amount"]
norm_boost  = [norm[c].mean()  for c in boost_cols]
smurf_boost = [smurf[c].mean() for c in boost_cols]

x3 = np.arange(len(boost_cats))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x3 - w/2, norm_boost,  w, label="Normal", color=NORMAL_COLOR)
ax.bar(x3 + w/2, smurf_boost, w, label="Smurf",  color=SMURF_COLOR)
ax.set_xticks(x3)
ax.set_xticklabels(boost_cats)
ax.set_ylabel("Boost units / amount")
ax.set_title("Boost Management: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 16: Pitch thirds grouped bar ─────────────────────────────────────────
third_cats   = ["Defensive third", "Neutral third", "Offensive third"]
third_cols   = ["percentage defensive third", "percentage neutral third", "percentage offensive third"]
norm_thirds  = [norm[c].mean()  for c in third_cols]
smurf_thirds = [smurf[c].mean() for c in third_cols]

x4 = np.arange(len(third_cats))
fig, ax = plt.subplots(figsize=(5, 7))
ax.bar(x4 - w/2, norm_thirds,  w, label="Normal", color=NORMAL_COLOR)
ax.bar(x4 + w/2, smurf_thirds, w, label="Smurf",  color=SMURF_COLOR)
ax.set_xticks(x4)
ax.set_xticklabels(third_cats)
ax.set_ylabel("% of game time")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Pitch Third Positioning: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 17: Top 5 smurf detection thresholds — one subplot each ─────────────
# Ranked by % of smurfs that exceed the normal player 90th percentile
threshold_stats = [
    "Score\n(90.3% above norm p90)",
    "Goals\n(90.3% above norm p90)",
    "Shots\n(80.6% above norm p90)",
    "Big pads collected\n(58.1% above norm p90)",
    "Time in offensive half\n(58.1% above norm p90)",
]
thresh_cols = [
    "score",
    "goals",
    "shots",
    "amount collected big pads",
    "time offensive half",
]
 
x_pos = np.arange(3)
bar_labels = ["Normal\nmedian", "Normal\n90th pct", "Smurf\nmedian"]
bar_colors = [NORMAL_COLOR, NORMAL_COLOR, SMURF_COLOR]
bar_alphas  = [0.6, 1.0, 1.0]
 
fig, axes = plt.subplots(1, 5, figsize=(18, 6))
fig.suptitle(
    "Top 5 Smurf Detection Thresholds\n(ranked by % of smurfs exceeding normal 90th percentile)",
    fontsize=12
)
 
for i, (col, title, ax) in enumerate(zip(thresh_cols, threshold_stats, axes)):
    values = [
        round(norm[col].median(), 1),
        round(norm[col].quantile(0.9), 1),
        round(smurf[col].median(), 1),
    ]
    bars = ax.bar(x_pos, values, width=0.6,
                  color=bar_colors,
                  alpha=1.0)
    bars[0].set_alpha(0.6)
 
    # Value labels above each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:g}",
            ha="center", va="bottom", fontsize=9,
        )
 
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_ylabel("Value" if i == 0 else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Give a little headroom above the tallest bar
    ax.set_ylim(0, max(values) * 1.18)
 
# Shared legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=NORMAL_COLOR, alpha=0.6, label="Normal median"),
    Patch(facecolor=NORMAL_COLOR, label="Normal 90th pct"),
    Patch(facecolor=SMURF_COLOR,  label="Smurf median"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.05), fontsize=9)

plt.subplots_adjust(bottom=0.15, top=0.85)  # Adjust to fit legend and title
plt.tight_layout()
plt.show()
 

# ── New plots ─────────────────────────────────────────────────────────────────

# Force numeric on new cols needed
for col in ["shots conceded", "goals conceded", "time offensive third",
            "amount stolen big pads", "count stolen big pads"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

norm  = df[df["smurf"] == 0]
smurf_df = df[df["smurf"] == 1]

rank_order = [
    "bronze-1","bronze-2","bronze-3",
    "silver-1","silver-2","silver-3",
    "gold-1","gold-2","gold-3",
    "platinum-1","platinum-2","platinum-3",
    "diamond-1","diamond-2","diamond-3",
    "champion-1","champion-2","champion-3",
    "grand-champion-1","grand-champion-2","grand-champion-3",
    "supersonic-legend",
]
rank_labels = [
    "B1","B2","B3","S1","S2","S3","G1","G2","G3",
    "P1","P2","P3","D1","D2","D3","C1","C2","C3",
    "GC1","GC2","GC3","SSL",
]

# ── Plot 18: Score vs goals scatter — smurf cluster ───────────────────────────
# Smurfs have a completely separate score/goals cluster vs normal players
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(norm["goals"], norm["score"],
           color=NORMAL_COLOR, alpha=0.2, s=12, label="Normal")
ax.scatter(smurf_df["goals"], smurf_df["score"],
           color=SMURF_COLOR, alpha=0.85, s=40, label="Smurf", zorder=5)
ax.set_xlabel("Goals scored")
ax.set_ylabel("Score")
ax.set_title("Score vs Goals: Smurfs Cluster Separately from Normal Players")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 19: Score delta vs rank expected ─────────────────────────────────────
# For each rank, compute the normal avg score, then show how far above it smurfs are
ranks_with_smurfs = [r for r in rank_order if len(smurf_df[smurf_df["rank"] == r]) > 0]
norm_avg_by_rank  = [norm[norm["rank"] == r]["score"].mean() for r in ranks_with_smurfs]
smurf_avg_by_rank = [smurf_df[smurf_df["rank"] == r]["score"].mean() for r in ranks_with_smurfs]
deltas = [s - n for s, n in zip(smurf_avg_by_rank, norm_avg_by_rank)]
short_labels = [r.replace("bronze","B").replace("silver","S").replace("gold","G")
                 .replace("platinum","P").replace("diamond","D").replace("champion","C")
                 .replace("grand-champion","GC").replace("supersonic-legend","SSL")
                 .replace("-","") for r in ranks_with_smurfs]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(short_labels, deltas, color=SMURF_COLOR, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, deltas):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"+{val:.0f}", ha="center", va="bottom", fontsize=8, color=SMURF_COLOR)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Rank (where smurfs were found)")
ax.set_ylabel("Score above rank average")
ax.set_title("How Far Above Rank Average Smurfs Score\n(every smurf outscores their rank's normal avg by 400–1,200 points)")
plt.tight_layout()
plt.show()

# ── Plot 20: Goals conceded comparison ────────────────────────────────────────
# Smurfs concede dramatically fewer goals — they dominate defensively too
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: box plot
sns.boxplot(data=df, x="smurf", y="goals conceded",
            palette=[NORMAL_COLOR, SMURF_COLOR], ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Normal", "Smurf"])
axes[0].set_title("Goals Conceded Distribution")
axes[0].set_ylabel("Goals conceded")

# Right: % with < 2 goals conceded
pct_norm  = (norm["goals conceded"] < 2).mean() * 100
pct_smurf = (smurf_df["goals conceded"] < 2).mean() * 100
axes[1].bar(["Normal", "Smurf"], [pct_norm, pct_smurf],
            color=[NORMAL_COLOR, SMURF_COLOR], alpha=0.85, edgecolor="white")
axes[1].set_ylabel("% of games")
axes[1].set_title("% of Games With < 2 Goals Conceded")
for i, val in enumerate([pct_norm, pct_smurf]):
    axes[1].text(i, val + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
axes[1].set_ylim(0, 100)

fig.suptitle("Defensive Dominance: Smurfs Concede Far Fewer Goals", fontsize=12)
plt.tight_layout()
plt.show()

# ── Plot 21: Offensive aggression fingerprint ─────────────────────────────────
# Combine % behind ball, time in offensive third, and shots into one grouped bar
aggression_cats = ["% behind ball", "% offensive third", "Shots per game"]
aggression_cols = ["percentage behind ball", "percentage offensive third", "shots"]
norm_agg  = [norm[c].mean()  for c in aggression_cols]
smurf_agg = [smurf_df[c].mean() for c in aggression_cols]

x = np.arange(len(aggression_cats))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
bars_n = ax.bar(x - w/2, norm_agg,  w, label="Normal", color=NORMAL_COLOR)
bars_s = ax.bar(x + w/2, smurf_agg, w, label="Smurf",  color=SMURF_COLOR)
for bar in list(bars_n) + list(bars_s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(aggression_cats)
ax.set_title("Offensive Aggression Fingerprint: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot 22: Big pad stealing — a skill that separates ranks ──────────────────
# Smurfs steal big pads at a much higher rate — a high-skill mechanical habit
fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(norm["amount stolen big pads"].dropna(),
            fill=True, label="Normal", color=NORMAL_COLOR)
sns.kdeplot(smurf_df["amount stolen big pads"].dropna(),
            fill=True, label="Smurf", color=SMURF_COLOR, alpha=0.7)
norm_med  = norm["amount stolen big pads"].median()
smurf_med = smurf_df["amount stolen big pads"].median()
ax.axvline(norm_med,  color=NORMAL_COLOR, linestyle="--", linewidth=1.5,
           label=f"Normal median ({norm_med:.0f})")
ax.axvline(smurf_med, color=SMURF_COLOR,  linestyle="--", linewidth=1.5,
           label=f"Smurf median ({smurf_med:.0f})")
ax.set_xlabel("Big boost pads stolen")
ax.set_title("Big Pad Stealing Distribution: Normal vs Smurf\n(smurfs deny boost at a much higher rate — a high-skill habit)")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plots 23–30: one plot per stat across Beginner / Intermediate / Expert / Smurf ──
smurf_stat_cols = [
    "avg speed",
    "amount used while supersonic",
    "time high in air",
    "percentage on ground",
    "percentage offensive third",
    "time supersonic speed",
    "avg powerslide time",
    "percentage behind ball",
]
smurf_stat_labels = [
    "Average Speed",
    "Boost Used While Supersonic",
    "Time High in Air",
    "Percentage on Ground",
    "Percentage Offensive Third",
    "Time Supersonic Speed",
    "Avg Powerslide Time",
    "Percentage Behind Ball",
]

for col in smurf_stat_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

tier_order  = ["Beginner", "Intermediate", "Expert", "Smurf"]
tier_colors = {
    "Beginner":     "#4CAF50",
    "Intermediate": "#FF9800",
    "Expert":       "#9C27B0",
    "Smurf":        SMURF_COLOR,
}

x     = np.arange(len(tier_order))
bar_w = 0.5

for col, label in zip(smurf_stat_cols, smurf_stat_labels):
    values = [df[df["tier"] == t][col].mean() for t in tier_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, values, width=bar_w,
                  color=[tier_colors[t] for t in tier_order],
                  edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontsize=11)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f"{label}: Beginner / Intermediate / Expert / Smurf", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()