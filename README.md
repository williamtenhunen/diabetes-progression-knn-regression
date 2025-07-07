# From Data to Insights: A KNN Regression Study on a Classic Diabetes Dataset

## Project Overview

This repository hosts the code and resources for a machine learning project focused on predicting diabetes disease progression. Leveraging the well-established load_diabetes dataset from scikit-learn, this study explores the application of the K-Nearest Neighbors (KNN) regression algorithm. The project emphasizes a comprehensive data analysis pipeline, from in-depth Exploratory Data Analysis (EDA) and data preprocessing to hyperparameter tuning and model evaluation.

Key Highlights:
* Exploratory Data Analysis (EDA): Deep dive into dataset characteristics, feature distributions, pairwise relationships, and dimensionality reduction using PCA.
* Feature Importance Analysis: Quantitative assessment of feature influence on diabetes progression using Linear Regression.
* Data Preprocessing: Implementation of essential steps including feature scaling and dataset splitting for robust model training and evaluation.
* KNN Regression Model: Construction, hyperparameter tuning (optimal k selection via Grid Search with Cross-Validation), and performance evaluation of the KNN model.
* Detailed Evaluation: Assessment of model accuracy and generalization using metrics like MSE, RMSE, MAE, and R-squared, with visual analysis of predictions.

## Medium Article

For a comprehensive walkthrough, detailed explanations, and the full narrative of this project, please refer to the accompanying Medium article:
[From Data to Insights: A KNN Regression Study on a Classic Diabetes Dataset](https://medium.com/@williamtenhunen/from-data-to-insights-a-knn-regression-study-on-a-classic-diabetes-dataset-d37aaf853314).

## Repository Structure
```
.
├── data/
│   └── diabetes-data.txt             # Original diabetes dataset (tab-separated)
├── src/
│   ├── eda_histograms.py             # Script for generating feature histograms
│   ├── eda_linear_regression.py      # Script for linear regression and feature importance
│   ├── eda_pca.py                    # Script for Principal Component Analysis (PCA)
│   ├── eda_scatter_plots.py          # Script for generating scatter plots
│   ├── eda_summary_statistics.py     # Script for generating summary statistics
│   └── knn_regression_model.py       # Script for KNN model training, tuning, and evaluation
├── images/                           # Directory for saving generated plots and visualizations
├── README.md                         # Project README file
├── LICENSE                           # Project LICENSE file
├── requirements.txt                  # List of Python dependencies
└── .gitignore                        # Specifies intentionally untracked files to ignore

```

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/williamtenhunen/diabetes-progression-knn-regression.git
pip install -r requirements.txt
```

## Technologies Used

* Python
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `statsmodels`
* `seaborn`
* Google Colab
* VS Code

## Author

William Tenhunen

## License

This project is licensed under the MIT License.
