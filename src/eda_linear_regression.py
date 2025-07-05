import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

file_path = 'diabetes-data.txt'

try:
  df = pd.read_csv(file_path, sep='\t')

  # Prepare data for linear regression
  features = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
  X = df[features]
  y = df['Y']

  # Standardise the features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Convert scaled features back to a DataFrame
  X_scaled_df = pd.DataFrame(X_scaled, columns=features)

  # Fit a simple linear regression
  model = LinearRegression()
  model.fit(X_scaled_df, y)

  # Extract and display coefficients
  coefficients = pd.DataFrame({
      'Feature': features,
      'Coefficient': model.coef_
  })

  # Sort by absolute coefficient value
  coefficients['Absolute Coefficient'] = abs(coefficients['Coefficient'])
  coefficients = coefficients.sort_values(by='Absolute Coefficient', ascending=False)

  print("Linear Regression Coefficients (Sorted by Absolute Magnitude)")
  print(coefficients)
  print("\n")

  # Visualize feature importance (Coefficients)
  sns.set_style('whitegrid')
  plt.style.use('seaborn-v0_8-pastel')

  plt.figure(figsize=(10, 7))
  sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
  plt.title('Feature Importance (Coefficients from Linear Regression)', fontsize=16, fontweight='bold')
  plt.xlabel('Coefficient Value', fontsize=12)
  plt.ylabel('Feature', fontsize=12)
  plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
  plt.grid(axis='x', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")