import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load with header row included
df = pd.read_csv("new_data.csv")

# Clean and convert types
df["X"] = df["X"].astype(int)
df["Y"] = df["Y"].astype(int)
df["Accuracy"] = df["Accuracy"].str.replace('%', '').astype(float)


# 2. Streamlit UI
st.title("Semi-Circular Player Heatmap")
selected_player = st.selectbox("Choose a player", df["Player"].unique())

# 3. Filter data
data = df[df["Player"] == selected_player]

# 4. Polar transformation
theta_map = dict(zip(range(1, 9), np.linspace(0, np.pi, 8)))
radius_map = dict(zip(range(1, 6), np.linspace(1, 5, 5)))
data["theta"] = data["X"].map(theta_map)
data["radius"] = data["Y"].map(radius_map)
data["x"] = data["radius"] * np.cos(data["theta"])
data["y"] = data["radius"] * np.sin(data["theta"])

# 5. Create polar mesh grid
theta = np.linspace(0, np.pi, 300)
radii = np.linspace(0, 5, 150)
theta_grid, radius_grid = np.meshgrid(theta, radii)
x = radius_grid * np.cos(theta_grid)
y = radius_grid * np.sin(theta_grid)

# 6. Interpolate accuracy values
interp = griddata((data["x"], data["y"]), data["Accuracy"], (x, y), method='cubic')
mask = radius_grid <= 5
masked = np.where(mask, interp, np.nan)

# 7. Plot
fig, ax = plt.subplots(figsize=(8, 8))
c = ax.contourf(x, y, masked, levels=100, cmap='coolwarm', vmin=0, vmax=100)
ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(c, ax=ax, label="Accuracy (%)")
plt.title(f"Heatmap: {selected_player}", fontsize=18)

st.pyplot(fig)
