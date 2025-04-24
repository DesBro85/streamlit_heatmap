import pandas as pd

# Load file with all rows as single strings
df_raw = pd.read_csv("your_data.csv", header=None)

# Split on commas
df_split = df_raw[0].str.split(",", expand=True)

# Drop the first row (it's a duplicate header)
df_split = df_split[1:]  # 🧼 Drop header row manually

# Rename columns
df_split.columns = ["X", "Y", "Accuracy", "Player"]

# Convert types safely
df_split["X"] = df_split["X"].astype(int)
df_split["Y"] = df_split["Y"].astype(int)
df_split["Accuracy"] = df_split["Accuracy"].str.replace('%', '').astype(float)

# Preview
print(df_split.dtypes)
print(df_split.head())
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Use your cleaned dataframe: df_split

# Step 1: Convert to polar coordinates
theta_map = dict(zip(range(1, 9), np.linspace(0, np.pi, 8)))
radius_map = dict(zip(range(1, 6), np.linspace(1, 5, 5)))
df_split['theta'] = df_split['X'].map(theta_map)
df_split['radius'] = df_split['Y'].map(radius_map)
df_split['x'] = df_split['radius'] * np.cos(df_split['theta'])
df_split['y'] = df_split['radius'] * np.sin(df_split['theta'])

# Step 2: Create polar mesh grid
theta = np.linspace(0, np.pi, 300)
radii = np.linspace(0, 5, 150)
theta_grid, radius_grid = np.meshgrid(theta, radii)
x = radius_grid * np.cos(theta_grid)
y = radius_grid * np.sin(theta_grid)

# Step 3: Interpolate
accuracy_interp = griddata(
    (df_split['x'], df_split['y']),
    df_split['Accuracy'],
    (x, y),
    method='cubic'
)

# Step 4: Mask the circular boundary
mask = radius_grid <= 5
masked_accuracy = np.where(mask, accuracy_interp, np.nan)

# Step 5: Plot it
fig, ax = plt.subplots(figsize=(10, 10))
c = ax.contourf(x, y, masked_accuracy, levels=100, cmap='coolwarm', vmin=0, vmax=100)
ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(c, ax=ax, label='Accuracy (%)')
plt.title("Heatmap (Full Dataset)", fontsize=20)
plt.show()
