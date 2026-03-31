import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/sw33t0404/Desktop/ISTA 498/ISTA_498_Capstone/replay_csvs/master_replays.csv", sep=',', low_memory=False)

# Clean smurf column (ensure 0/1 integers)
df["smurf"] = df["smurf"].fillna(0).astype(int)

# Distribution of rank plot
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="rank", hue="smurf", palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Rank Distribution: Smurfs vs Normal Players")
plt.tight_layout()
plt.show()

# Avg score comparison
score_comp = df.groupby("smurf")["score"].mean()

plt.bar(["Normal","Smurf"], score_comp, color=["blue","red"])
plt.ylabel("Average Score")
plt.title("Average Score: Normal vs Smurfs")
plt.show()

# Goals, Assists, Saves - Radar Chart
import numpy as np

stats = ["goals","assists","saves"]
vals_norm = df[df.smurf==0][stats].mean().values
vals_smurf = df[df.smurf==1][stats].mean().values

angles = np.linspace(0, 2*np.pi, len(stats), endpoint=False)
vals_norm = np.concatenate((vals_norm,[vals_norm[0]]))
vals_smurf = np.concatenate((vals_smurf,[vals_smurf[0]]))
angles = np.concatenate((angles,[angles[0]]))

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, vals_norm, label="Normal", linewidth=2)
ax.plot(angles, vals_smurf, label="Smurf", linewidth=2)
ax.fill(angles, vals_norm, alpha=0.2)
ax.fill(angles, vals_smurf, alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(stats)
plt.title("Radar Chart: Gameplay Stats Comparison")
plt.legend()
plt.show()

# Heatmap of Correlation for Smurfs vs Non-Smurfs
plt.figure(figsize=(10,6))
corr_smurf = df[df.smurf==1][["score","goals","assists","saves","shots","avg speed"]].corr()
sns.heatmap(corr_smurf, annot=True, cmap="Reds")
plt.title("Smurf Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,6))
corr_norm = df[df.smurf==0][["score","goals","assists","saves","shots","avg speed"]].corr()
sns.heatmap(corr_norm, annot=True, cmap="Blues")
plt.title("Normal Player Correlation Heatmap")
plt.show()

# Smurfs vs Normal - Speed distribution
plt.figure(figsize=(10,5))
sns.kdeplot(df[df.smurf==0]["avg speed"], fill=True, label="Normal")
sns.kdeplot(df[df.smurf==1]["avg speed"], fill=True, label="Smurf")
plt.title("Speed Distribution: Normal vs Smurf")
plt.xlabel("Average Speed")
plt.legend()
plt.show()

# Boost usage difference
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="smurf", y="avg boost amount", palette="coolwarm")
plt.xticks([0,1], ["Normal","Smurf"])
plt.title("Boost Usage Comparison")
plt.show()

# Time behind ball vs in front
plt.scatter(
    df["time behind ball"],
    df["time in front of ball"],
    c=df["smurf"],
    cmap="coolwarm",
    alpha=0.3,
    s=10
)

plt.xlabel("Time Behind Ball")
plt.ylabel("Time In Front of Ball")
plt.title("Field Positioning: Smurfs vs Normal")
plt.show()