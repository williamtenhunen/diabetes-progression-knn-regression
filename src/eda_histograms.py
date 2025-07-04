import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'diabetes-data.txt'

try:
  # Load the diabetes data
  df = pd.read_csv(file_path, sep='\t')

  # A professional looking palette for the visualization
  sns.set_style('whitegrid')
  plt.style.use("seaborn-v0_8-pastel")

  numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

  # Number of rows and cols for the subplot grid
  n_cols = 3
  n_rows = (len(numerical_cols) + n_cols - 1)

  # Subplots for all histograms
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

  # Flatten the 2D array of axes
  axes = axes.flatten()

  # Generate a histogram for each numerical column
  for i, col in enumerate(numerical_cols):
    if i < len(axes):
      sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
      axes[i].set_title(f'Distribution of {col}', fontsize=14)
      axes[i].set_xlabel(col, fontsize=12)
      axes[i].set_ylabel('Frequency', fontsize=12)

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  # Adjust layout
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])

  # A global title
  plt.suptitle('Histograms of Diabetes Dataset Features', y=1.00, fontsize=16, fontweight='bold')

  plt.show()

except FileNotFoundError:
  print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")