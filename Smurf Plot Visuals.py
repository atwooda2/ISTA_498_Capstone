import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

df = pd.read_csv("/Users/sw33t0404/Desktop/ISTA 498/ISTA_498_Capstone/replay_csvs/master_replays.csv", low_memory=False)

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
SMURF_COLOR  = "#E24B4A"

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
    "champion-1": "Advanced",      "champion-2": "Advanced",      "champion-3": "Advanced",
    "grand-champion-1": "Advanced","grand-champion-2": "Advanced","grand-champion-3": "Advanced",
    "supersonic-legend": "Advanced",
}
df["tier"] = df["rank"].map(tier_map)
df.loc[df["smurf"] == 1, "tier"] = "Smurf"


# ── Plot 1: Rank distribution ─────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="rank", hue="smurf", palette="coolwarm", order=rank_order)
plt.xticks(ticks=range(len(rank_labels)), labels=rank_labels, rotation=45)
plt.title("Rank Distribution: Smurfs vs Normal Players")
plt.tight_layout()
plt.show()

# ── Plot 2: Avg score comparison ──────────────────────────────────────────────
score_comp = df.groupby("smurf")["score"].mean()
plt.bar(["Normal", "Smurf"], score_comp, color=[NORMAL_COLOR, SMURF_COLOR])
plt.ylabel("Average Score")
plt.title("Average Score: Normal vs Smurfs")
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
    "Advanced":     {"color": "#3F51B5",   "alpha": 0.35, "s": 10},
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
fig, ax = plt.subplots(figsize=(8, 5))
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

# ── Plot 17: Top 5 smurf detection thresholds ────────────────────────────────
# Ranked by % of smurfs that exceed the normal player 90th percentile
threshold_stats = [
    "Score\n(90.3% of smurfs\nabove norm p90)",
    "Goals\n(90.3% of smurfs\nabove norm p90)",
    "Shots\n(80.6% of smurfs\nabove norm p90)",
    "Big pad\ncollected\n(58.1% above)",
    "Time in\noffensive half\n(58.1% above)",
]
thresh_cols = [
    "score",
    "goals",
    "shots",
    "amount collected big pads",
    "time offensive half",
]
 
normal_medians = [round(norm[c].median(), 1) for c in thresh_cols]
normal_p90     = [round(norm[c].quantile(0.9), 1) for c in thresh_cols]
smurf_medians  = [round(smurf[c].median(), 1) for c in thresh_cols]
 
x5 = np.arange(len(threshold_stats))
fig, ax = plt.subplots(figsize=(13, 7))
bars_n  = ax.bar(x5 - w, normal_medians, w, label="Normal median",   color=NORMAL_COLOR, alpha=0.6)
bars_p  = ax.bar(x5,     normal_p90,     w, label="Normal 90th pct", color=NORMAL_COLOR)
bars_s  = ax.bar(x5 + w, smurf_medians,  w, label="Smurf median",    color=SMURF_COLOR)
 
# Value labels on top of each bar
for bar in bars_n:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{bar.get_height():g}", ha="center", va="bottom", fontsize=8, color=NORMAL_COLOR)
for bar in bars_p:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{bar.get_height():g}", ha="center", va="bottom", fontsize=8, color=NORMAL_COLOR)
for bar in bars_s:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{bar.get_height():g}", ha="center", va="bottom", fontsize=8, color=SMURF_COLOR)
 
ax.set_xticks(x5)
ax.set_xticklabels(threshold_stats, ha="center", fontsize=9)
ax.set_ylabel("Stat value")
ax.set_title("Top 5 Smurf Detection Thresholds\n(ranked by % of smurfs exceeding normal 90th percentile)")
ax.legend()
plt.tight_layout()
plt.show()
 