import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load with header row included
df = pd.read_csv("new_data.csv")

# Get file last modified timestamp
file_path = "new_data.csv"
last_modified = os.path.getmtime(file_path)
last_updated = datetime.datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")

# Clean and convert types
df = pd.read_csv("new_data.csv")
df["X"] = df["X"].astype(int)
df["Y"] = df["Y"].astype(int)
df["Percentage"] = df["Percentage"].str.replace('%', '').astype(float)

# 2. Streamlit UI
st.title("Shooting Heatmap")
selected_player = st.selectbox("Choose Player", df["Player"].unique())
selected_period = st.selectbox("Choose Period", df["Period"].unique())
selected_show = st.selectbox("Choose Show Type", df["Show"].unique())

# Show last updated time
st.markdown(f"🕒 **Last updated:** {last_updated}")

# 3. Filter data
data = df[
    (df["Player"] == selected_player) &
    (df["Period"] == selected_period) &
    (df["Show"] == selected_show)].copy()



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
interp = griddata(
    (data['x'], data['y']),    data['Percentage'],    (x, y),    method='cubic')
interp = np.clip(interp, 0, 100)
mask = radius_grid <= 5
masked = np.where(mask, interp, np.nan)

# Check if filtered data has any rows
if len(data) == 0 or data["Percentage"].dropna().sum() == 0:
    st.warning("⚠️ No data available for this selection.")
else:
    # Proceed to build heatmap
    vmin = 0
    vmax = data["Percentage"].max()

    if vmax < 1e-3:  # Protect against broken scale
        vmax = 1

# Create polar grid etc...
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.contourf(x, y, masked, levels=100, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(c, ax=ax, label="Percentage (%)")
    plt.title(f"Heatmap: {selected_player} | {selected_period} | {selected_show}")

    st.pyplot(fig)


# Commit to Git
# Trigger redeploy: Minor comment update
