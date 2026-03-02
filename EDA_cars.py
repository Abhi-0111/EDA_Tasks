import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the dataset (Use the exact name of your file)
file_path = 'EDA_Cars.xlsx' 
df = pd.read_excel(file_path)

# --- 2. Data Cleaning ---
# Drop 'INDEX' if it exists
if 'INDEX' in df.columns:
    df = df.drop(columns=['INDEX'])

# Handle missing values for numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle missing values for categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- 3. Visualization ---

# Distribution of Numerical Columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i+1) # Adjusted grid size based on column count
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(f'Dist of {col}')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Income vs Car Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='CAR TYPE', y='INCOME', palette='Set2')
plt.xticks(rotation=45)
plt.title('Income by Car Type')
plt.show()