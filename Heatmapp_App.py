import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Load your dataset (update the path if needed)
df = pd.read_csv("your_data.csv")  # or use pd.read_excel() if it's Excel

# Clean accuracy values
df['Accuracy'] = df['Accuracy'].str.replace('%', '').astype(float)

# Sidebar dropdown
player = st.sidebar.selectbox("Select Player", df["Player"].unique())

# Filter the data
data = df[df["Player"] == player]

# Convert to polar coordinates
theta_map = dict(zip(range(1, 9), np.linspace(0, np.pi, 8)))
radius_map = dict(zip(range(1, 6), np.linspace(1, 5, 5)))
data['theta'] = data['X'].map(theta_map)
data['radius'] = data['Y'].map(radius_map)
data['x'] = data['radius'] * np.cos(data['theta'])
data['y'] = data['radius'] * np.sin(data['theta'])

# Create polar grid
theta = np.linspace(0, np.pi, 300)
radii = np.linspace(0, 5, 150)
theta_grid, radius_grid = np.meshgrid(theta, radii)
x = radius_grid * np.cos(theta_grid)
y = radius_grid * np.sin(theta_grid)

# Interpolate accuracy values
interpolated = griddata((data['x'], data['y']), data['Accuracy'], (x, y), method='cubic')
mask = radius_grid <= 5
masked_accuracy = np.where(mask, interpolated, np.nan)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
c = ax.contourf(x, y, masked_accuracy, levels=100, cmap='coolwarm', vmin=0, vmax=100)
ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(c, ax=ax, label='Accuracy (%)')
plt.title(f"{player}", fontsize=18)

st.pyplot(fig)
