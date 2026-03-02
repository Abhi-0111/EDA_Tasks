import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('EDA Cars.xlsx - Sheet1.csv')

# --- 1. Data Cleaning ---
# Drop 'INDEX' as it is a unique identifier and not useful for analysis
df_clean = df.drop(columns=['INDEX'])

# Handle missing values:
# Numerical: fill with median (more robust to outliers than mean)
num_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Categorical: fill with mode (the most frequent value)
cat_cols = df_clean.select_dtypes(include=['object']).columns
for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# --- 2. Univariate Analysis (Numerical Distributions) ---
plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols):
    plt.subplot(3, 2, i+1)
    sns.histplot(df_clean[col], kde=True, color='teal')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numerical_distributions.png')

# --- 3. Univariate Analysis (Categorical Counts) ---
plt.figure(figsize=(15, 20))
for i, col in enumerate(cat_cols):
    plt.subplot(4, 2, i+1)
    sns.countplot(data=df_clean, x=col, palette='viridis', 
                  order=df_clean[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Frequency of {col}')
plt.tight_layout()
plt.savefig('categorical_distributions.png')

# --- 4. Bivariate Analysis (Correlation Heatmap) ---
plt.figure(figsize=(10, 6))
correlation_matrix = df_clean[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('correlation_heatmap.png')

# --- 5. Bivariate Analysis (Key Relationships) ---
# Relationship between Income and Car Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean, x='CAR TYPE', y='INCOME', palette='Set2')
plt.title('Income Distribution by Car Type')
plt.savefig('income_vs_cartype.png')

# Relationship between Vehicle Use and Travel Time
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_clean, x='USE', y='TRAVEL TIME', palette='Pastel1')
plt.title('Travel Time by Vehicle Use')
plt.savefig('travel_time_vs_use.png')

print("EDA completed and plots saved.")