import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: 10 time steps of heatmaps
time_steps = 10
angles = np.linspace(0, np.pi, 100)
radii = np.linspace(1, 6, 6)
angle_grid, radius_grid = np.meshgrid(angles, radii)
heatmaps = [np.random.rand(len(radii), len(angles)) for _ in range(time_steps)]

# Convert to Cartesian once
x = radius_grid * np.cos(angle_grid)
y = radius_grid * np.sin(angle_grid)

# Streamlit layout
st.title("Semi-Circle Heatmap with Time Slider")
time_index = st.slider("Select Time Step", 0, time_steps - 1, 0)

# Plot the selected heatmap
fig, ax = plt.subplots(figsize=(8, 4))
heat_values = heatmaps[time_index]
c = ax.pcolormesh(x, y, heat_values, shading='auto', cmap='hot')
ax.set_aspect('equal')
ax.axis('off')
fig.colorbar(c, ax=ax, orientation='horizontal', label='Heat Intensity')
plt.title(f"Time Step {time_index}")

# Show plot in Streamlit
st.pyplot(fig)