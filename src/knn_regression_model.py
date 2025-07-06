import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('diabetes-data.txt', sep='\t')

# Define features (X) and target (y)
features = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
X = df[features]
y = df['Y']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=101)

# Hyperparameter tuning for KNN
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': range(1, 31)}

# Setup GridSearchCV with 5-fold cross-validation and fit to the training data
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best 'k' and its score
best_k = grid_search.best_params_['n_neighbors']
best_score_mse = -grid_search.best_score_

print(f"Best 'k': {best_k}")
print(f"Best (lowest) Mean Squared Error during cross-validation: {best_score_mse:.2f}")

# Train the final KNN Model with the Optimal 'k'
final_knn_model = KNeighborsRegressor(n_neighbors=best_k)
final_knn_model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = final_knn_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation on Test Data\n")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Visualize actual vs predicted values ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Actual vs Predicted Values (KNN Regression)')
plt.tight_layout()
plt.show()