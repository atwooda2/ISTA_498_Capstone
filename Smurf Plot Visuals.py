import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

df = pd.read_csv("/Users/sw33t0404/Desktop/ISTA 498/ISTA_498_Capstone/replay_csvs/master_replays.csv", low_memory=False)

# Clean smurf column (ensure 0/1 integers)
df["smurf"] = df["smurf"].fillna(0).astype(int)

# ── shared palette ────────────────────────────────────────────────────────────
NORMAL_COLOR = "#378ADD"
SMURF_COLOR  = "#E24B4A"
NORMAL_LIGHT = "#378ADD55"
SMURF_LIGHT  = "#E24B4A55"

norm  = df[df["smurf"] == 0]
smurf = df[df["smurf"] == 1]


# ── existing plots ────────────────────────────────────────────────────────────

# Distribution of rank plot
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="rank", hue="smurf", palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Rank Distribution: Smurfs vs Normal Players")
plt.tight_layout()
plt.show()

# Avg score comparison
score_comp = df.groupby("smurf")["score"].mean()
plt.bar(["Normal", "Smurf"], score_comp, color=["blue", "red"])
plt.ylabel("Average Score")
plt.title("Average Score: Normal vs Smurfs")
plt.show()

# Goals, Assists, Saves - Radar Chart
stats = ["goals", "assists", "saves"]
vals_norm  = df[df.smurf == 0][stats].mean().values
vals_smurf = df[df.smurf == 1][stats].mean().values

angles     = np.linspace(0, 2 * np.pi, len(stats), endpoint=False)
vals_norm  = np.concatenate((vals_norm,  [vals_norm[0]]))
vals_smurf = np.concatenate((vals_smurf, [vals_smurf[0]]))
angles     = np.concatenate((angles,     [angles[0]]))

fig = plt.figure(figsize=(6, 6))
ax  = plt.subplot(111, polar=True)
ax.plot(angles, vals_norm,  label="Normal", linewidth=2)
ax.plot(angles, vals_smurf, label="Smurf",  linewidth=2)
ax.fill(angles, vals_norm,  alpha=0.2)
ax.fill(angles, vals_smurf, alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(stats)
plt.title("Radar Chart: Gameplay Stats Comparison")
plt.legend()
plt.show()

# Heatmap of Correlation for Smurfs vs Non-Smurfs
plt.figure(figsize=(10, 6))
corr_smurf = df[df.smurf == 1][["score", "goals", "assists", "saves", "shots", "avg speed"]].corr()
sns.heatmap(corr_smurf, annot=True, cmap="Reds")
plt.title("Smurf Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
corr_norm = df[df.smurf == 0][["score", "goals", "assists", "saves", "shots", "avg speed"]].corr()
sns.heatmap(corr_norm, annot=True, cmap="Blues")
plt.title("Normal Player Correlation Heatmap")
plt.show()

# Smurfs vs Normal - Speed distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(df[df.smurf == 0]["avg speed"], fill=True, label="Normal")
sns.kdeplot(df[df.smurf == 1]["avg speed"], fill=True, label="Smurf")
plt.title("Speed Distribution: Normal vs Smurf")
plt.xlabel("Average Speed")
plt.legend()
plt.show()

# Boost usage difference
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="smurf", y="avg boost amount", palette="coolwarm")
plt.xticks([0, 1], ["Normal", "Smurf"])
plt.title("Boost Usage Comparison")
plt.show()

# Time behind ball vs in front
plt.scatter(
    df["time behind ball"],
    df["time in front of ball"],
    c=df["smurf"],
    cmap="coolwarm",
    alpha=0.3,
    s=10,
)
plt.xlabel("Time Behind Ball")
plt.ylabel("Time In Front of Ball")
plt.title("Field Positioning: Smurfs vs Normal")
plt.show()


# ── new plots ─────────────────────────────────────────────────────────────────

# 1. Score distribution histogram
fig, ax = plt.subplots(figsize=(10, 5))
bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 2000]
ax.hist(norm["score"],  bins=bins, color=NORMAL_COLOR, alpha=0.6, label="Normal", edgecolor="white")
ax.hist(smurf["score"], bins=bins, color=SMURF_COLOR,  alpha=0.8, label="Smurf",  edgecolor="white")
ax.axvline(norm["score"].median(),  color=NORMAL_COLOR, linestyle="--", linewidth=1.5, label=f"Normal median ({int(norm['score'].median())})")
ax.axvline(smurf["score"].median(), color=SMURF_COLOR,  linestyle="--", linewidth=1.5, label=f"Smurf median ({int(smurf['score'].median())})")
ax.set_xlabel("Score")
ax.set_ylabel("Player count")
ax.set_title("Score Distribution: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# 2. Shooting % distribution histogram
fig, ax = plt.subplots(figsize=(10, 5))
bins_pct = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
ax.hist(pd.to_numeric(norm["shooting percentage"],  errors="coerce").dropna(),
        bins=bins_pct, color=NORMAL_COLOR, alpha=0.6, label="Normal", edgecolor="white")
ax.hist(pd.to_numeric(smurf["shooting percentage"], errors="coerce").dropna(),
        bins=bins_pct, color=SMURF_COLOR,  alpha=0.8, label="Smurf",  edgecolor="white")
ax.set_xlabel("Shooting percentage (%)")
ax.set_ylabel("Player count")
ax.set_title("Shooting % Distribution: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# 3. Speed profile grouped bar (slow / boost / supersonic %)
speed_cats   = ["Slow speed", "Boost speed", "Supersonic"]
speed_cols   = ["percentage slow speed", "percentage boost speed", "percentage supersonic speed"]
norm_means   = [pd.to_numeric(norm[c],  errors="coerce").mean() for c in speed_cols]
smurf_means  = [pd.to_numeric(smurf[c], errors="coerce").mean() for c in speed_cols]

x   = np.arange(len(speed_cats))
w   = 0.35
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

# 4. Air vs ground grouped bar
air_cats   = ["On ground", "Low air", "High air"]
air_cols   = ["percentage on ground", "percentage low in air", "percentage high in air"]
norm_air   = [pd.to_numeric(norm[c],  errors="coerce").mean() for c in air_cols]
smurf_air  = [pd.to_numeric(smurf[c], errors="coerce").mean() for c in air_cols]

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

# 5. Rank progression — avg speed across ranks with smurf reference line
rank_order = [
    "bronze-1","bronze-2","bronze-3",
    "silver-1","silver-2","silver-3",
    "gold-1","gold-2","gold-3",
    "platinum-1","platinum-2","platinum-3",
    "diamond-1","diamond-2","diamond-3",
    "champion-1","champion-2","champion-3",
    "grand-champion-1","supersonic-legend",
]
rank_labels = [
    "B1","B2","B3","S1","S2","S3","G1","G2","G3",
    "P1","P2","P3","D1","D2","D3","C1","C2","C3","GC1","SSL",
]

rank_speed = (
    df[df["rank"].isin(rank_order)]
    .groupby("rank")["avg speed"]
    .mean()
    .reindex(rank_order)
)
smurf_speed_avg = pd.to_numeric(smurf["avg speed"], errors="coerce").mean()

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

# 6. Rank progression — supersonic % across ranks with smurf reference line
rank_sup = (
    df[df["rank"].isin(rank_order)]
    .groupby("rank")["percentage supersonic speed"]
    .mean()
    .reindex(rank_order)
)
smurf_sup_avg = pd.to_numeric(smurf["percentage supersonic speed"], errors="coerce").mean()

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

# 7. Boost collected vs stolen — grouped bar
boost_cats  = ["Boost collected", "Boost stolen", "Avg boost amount"]
boost_cols  = ["amount collected", "amount stolen", "avg boost amount"]
norm_boost  = [pd.to_numeric(norm[c],  errors="coerce").mean() for c in boost_cols]
smurf_boost = [pd.to_numeric(smurf[c], errors="coerce").mean() for c in boost_cols]

x3  = np.arange(len(boost_cats))
fig, ax = plt.subplots(figsize=(8, 5))
bars_n = ax.bar(x3 - w/2, norm_boost,  w, label="Normal", color=NORMAL_COLOR)
bars_s = ax.bar(x3 + w/2, smurf_boost, w, label="Smurf",  color=SMURF_COLOR)
ax.set_xticks(x3)
ax.set_xticklabels(boost_cats)
ax.set_ylabel("Boost units / amount")
ax.set_title("Boost Management: Normal vs Smurf")
ax.legend()
plt.tight_layout()
plt.show()

# 8. Pitch thirds positioning grouped bar
third_cats  = ["Defensive third", "Neutral third", "Offensive third"]
third_cols  = ["percentage defensive third", "percentage neutral third", "percentage offensive third"]
norm_thirds  = [pd.to_numeric(norm[c],  errors="coerce").mean() for c in third_cols]
smurf_thirds = [pd.to_numeric(smurf[c], errors="coerce").mean() for c in third_cols]

x4  = np.arange(len(third_cats))
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

# 9. Detection threshold bar chart
threshold_stats  = ["Score", "Goals/game", "Shooting %", "High air %", "Supersonic %", "Boost stolen", "Offensive third %", "BPM"]
normal_medians   = [390,  1.0,  40.0,  2.8, 10.0, 300, 21.0, 336]
normal_p90       = [738,  3.0, 100.0,  6.5, 18.0, 620, 29.0, 452]
smurf_medians    = [1375, 8.1,  77.0,  6.6, 15.4, 675, 30.0, 424]

x5 = np.arange(len(threshold_stats))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x5 - w,   normal_medians, w, label="Normal median",  color=NORMAL_COLOR, alpha=0.6)
ax.bar(x5,       normal_p90,     w, label="Normal 90th pct",color=NORMAL_COLOR)
ax.bar(x5 + w,   smurf_medians,  w, label="Smurf median",   color=SMURF_COLOR)
ax.set_xticks(x5)
ax.set_xticklabels(threshold_stats, rotation=20, ha="right")
ax.set_ylabel("Stat value")
ax.set_title("Smurf Detection Thresholds: Normal vs Smurf Key Stats")
ax.legend()
plt.tight_layout()
plt.show()