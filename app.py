import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load and clean data
df_raw = pd.read_csv("your_data.csv", header=None)
df = df_raw[0].str.split(",", expand=True)
df = df[1:]  # Drop header row
df.columns = ["X", "Y", "Accuracy", "Player"]
df["X"] = df["X"].astype(int)
df["Y"] = df["Y"].astype(int)
df["Accuracy"] = df["Accuracy"].str.replace('%', '').astype(float)

# Streamlit title
st.title("Player Heatmap Viewer")

# Dropdown for player selection
players = df["Player"].unique()
selected_player = st.selectbox("Choose a player", players)

# Filter the data for the selected player
data = df[df["Player"] == selected_player]

# Polar transform
theta_map = dict(zip(range(1, 9), np.linspace(0, np.pi, 8)))
radius_map = dict(zip(range(1, 6), np.linspace(1, 5, 5)))
data["theta"] = data["X"].map(theta_map)
data["radius"] = data["Y"].map(radius_map)
data["x"] = data["radius"] * np.cos(data["theta"])
data["y"] = data["radius"] * np.sin(data["theta"])

# Polar grid
theta = np.linspace(0, np.pi, 300)
radii = np.linspace(0, 5, 150)
theta_grid, radius_grid = np.meshgrid(theta, radii)
x = radius_grid * np.cos(theta_grid)
y = radius_grid * np.sin(theta_grid)

# Interpolation
interp = griddata((data['x'], data['y']), data['Accuracy'], (x, y), method='cubic')
mask = radius_grid <= 5
masked = np.where(mask, interp, np.nan)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
c = ax.contourf(x, y, masked, levels=100, cmap='coolwarm', vmin=0, vmax=100)
ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(c, ax=ax, label='Accuracy (%)')
plt.title(f"Heatmap: {selected_player}", fontsize=18)

# Display in Streamlit
st.pyplot(fig)
