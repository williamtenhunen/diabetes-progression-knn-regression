import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'diabetes-data.txt'

try:
    df = pd.read_csv(file_path, sep='\t')

    # Set a professional style for seaborn plots
    sns.set_style("whitegrid")
    plt.style.use("seaborn-v0_8-pastel")

    # Define pairs of variables to visualize with scatter plots
    plot_pairs = [
        ('BMI', 'Y'),
        ('BP', 'Y'),
        ('AGE', 'Y'),
        ('S5', 'Y'),
        ('BMI', 'BP')
    ]

    # Determine grid size for subplots
    n_plots = len(plot_pairs)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    # Create scatter plots
    for i, (x_col, y_col) in enumerate(plot_pairs):
        if i < len(axes):
            # Adding 'hue='SEX'' to see if relationships differ by sex
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue='SEX',
                palette='deep',
                s=50,
                alpha=0.7,
                ax=axes[i]
            )
            axes[i].set_title(f'{x_col} vs. {y_col} (Colored by Sex)', fontsize=14)
            axes[i].set_xlabel(x_col, fontsize=12)
            axes[i].set_ylabel(y_col, fontsize=12)
            axes[i].legend(title='SEX')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.suptitle('Exploring Relationships with Scatter Plots', y=1.00, fontsize=18, fontweight='bold')
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")