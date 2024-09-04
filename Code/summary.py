import pandas as pd

# Load the dataset
file_path = 'All_Data_with_features.csv'
data = pd.read_csv(file_path)

# Grouping data by CEFR level and calculating mean and standard deviation for each feature
grouped_data = data.groupby('Cefr').agg(
    Grammar_Errors_Mean=('Grammar_Errors', 'mean'),
    Grammar_Errors_Std=('Grammar_Errors', 'std'),
    Type_Token_Ratio_Mean=('Type_Token_Ratio', 'mean'),
    Type_Token_Ratio_Std=('Type_Token_Ratio', 'std'),
    Lexical_Density_Mean=('Lexical_Density', 'mean'),
    Lexical_Density_Std=('Lexical_Density', 'std'),
    Flesch_Reading_Ease_Mean=('Flesch_Reading_Ease', 'mean'),
    Flesch_Reading_Ease_Std=('Flesch_Reading_Ease', 'std'),
    Wiener_Sachtextformel_Mean=('Wiener_Sachtextformel_Average', 'mean'),
    Wiener_Sachtextformel_Std=('Wiener_Sachtextformel_Average', 'std')
).reset_index()
