import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Define the data from your image
data_test2 = {
    "X": [1,2,3,4,5,6,7,8]*5,
    "Y": [1]*8 + [2]*8 + [3]*8 + [4]*8 + [5]*8,
    "Accuracy": [
        100, 100, 90, 83, 80, 100, 100, 100,      # Y = 1
        100, 100, 0, 0, 0, 83, 100, 100,          # Y = 2
        0, 0, 0, 0, 0, 0, 0, 0,                   # Y = 3
        0, 0, 0, 0, 0, 0, 0, 0,                   # Y = 4
        0, 0, 0, 0, 20, 0, 0, 0                   # Y = 5
    ]
}
df = pd.DataFrame(data_test2)

# Map to polar coordinates
theta_map = dict(zip(range(1, 9), np.linspace(0, np.pi, 8)))
radius_map = dict(zip(range(1, 6), np.linspace(1, 5, 5)))
df['theta'] = df['X'].map(theta_map)
df['radius'] = df['Y'].map(radius_map)
df['x'] = df['radius'] * np.cos(df['theta'])
df['y'] = df['radius'] * np.sin(df['theta'])

# Create a high-res polar meshgrid
theta = np.linspace(0, np.pi, 300)
radii = np.linspace(0, 5, 150)
theta_grid, radius_grid = np.meshgrid(theta, radii)
x = radius_grid * np.cos(theta_grid)
y = radius_grid * np.sin(theta_grid)

# Interpolate the accuracy values over the grid
accuracy_interp = griddata(
    (df['x'], df['y']),
    df['Accuracy'],
    (x, y),
    method='cubic'
)

# Apply a circular mask to create a curved outer edge
mask = radius_grid <= 5
masked_accuracy = np.where(mask, accuracy_interp, np.nan)

# Plotting the final visual
fig, ax = plt.subplots(figsize=(10, 10))  # Enlarged figure
c = ax.contourf(x, y, masked_accuracy, levels=100, cmap='coolwarm', vmin=0, vmax=100)  # Color range capped at 100%
ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(c, ax=ax, label='Accuracy (%)')
plt.title("Test 2", fontsize=20)

plt.show()
