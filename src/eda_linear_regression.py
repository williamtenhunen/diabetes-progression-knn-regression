import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

file_path = 'diabetes-data.txt'

try:
    df = pd.read_csv(file_path, sep='\t')

    # Prepare Data for Linear Regression
    features = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    X = df[features]
    y = df['Y']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled features back to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # Fit Model using sklearn
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_scaled_df, y)

    coefficients_sklearn = pd.DataFrame({
        'Feature': features,
        'Coefficient': sklearn_model.coef_
    })
    coefficients_sklearn['Absolute Coefficient'] = abs(coefficients_sklearn['Coefficient'])
    coefficients_sklearn = coefficients_sklearn.sort_values(by='Absolute Coefficient', ascending=False)

    print("Sklearn Linear Regression Coefficients (Sorted by Absolute Magnitude)")
    print(coefficients_sklearn)
    print("\n")

    # Fit Model using statsmodels
    X_scaled_df_with_const = sm.add_constant(X_scaled_df)

    # Create and fit the OLS (Ordinary Least Squares) model
    statsmodel_model = sm.OLS(y, X_scaled_df_with_const)
    statsmodel_results = statsmodel_model.fit()

    # Print the full summary table
    print("Statsmodels Linear Regression Summary (includes t-values and p-values)")
    print(statsmodel_results.summary())
    print("\n")

    # Visualize Feature Importance (Coefficients)
    sns.set_style("whitegrid")
    plt.style.use("seaborn-v0_8-pastel")

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients_sklearn, palette='viridis')
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