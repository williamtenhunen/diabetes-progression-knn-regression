import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_path = 'diabetes-data.txt'

try:
    df = pd.read_csv(file_path, sep='\t')

    # Select features for PCA.
    features = ['AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    X = df[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce to 2 principal components for 2D visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

    # Add back the 'Y' (disease progression) and 'SEX' columns
    pca_df['Disease Progression (Y)'] = df['Y']
    pca_df['SEX'] = df['SEX'].map({1: 'Male', 2: 'Female'}) # Map SEX to labels for better legend

    # Concatenate the original features DataFrame to pca_df
    pca_df = pd.concat([pca_df, df[features]], axis=1)

    print("Head of pca_df after adding original features")
    print(pca_df.head())
    print("\n")

    # Create Interactive PCA Plot with Plotly
    fig = px.scatter(
        pca_df,
        x='Principal Component 1',
        y='Principal Component 2',
        color='Disease Progression (Y)',
        hover_data=['Disease Progression (Y)', 'SEX'] + features,
        title='2D PCA of Diabetes Dataset Features (Colored by Disease Progression)',
        labels={'Principal Component 1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                'Principal Component 2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'},
        height=600
    )

    fig.update_layout(
        title_font_size=20,
        hovermode="closest"
    )

    # Display the plot
    fig.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")