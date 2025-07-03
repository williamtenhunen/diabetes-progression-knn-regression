import pandas as pd

file_path = 'diabetes-data.txt'

try:
  df = pd.read_csv(file_path, sep='\t') # Using the original Stanford dataset
  
  print("Dataset Information")
  df.info()
  print("\n")

  print("Descriptive Statistics")
  print(df.describe())

except FileNotFoundError:
  print(f"Error: The file {file_path} was not found.")
  print("Please ensure the 'diabetes-data.txt' file is in the correct directory.")
except Exception as e:
  print(f"An error occurred: {e}")