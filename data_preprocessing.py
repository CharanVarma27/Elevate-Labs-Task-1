import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("Step 1: Importing and Exploring the Dataset\n")
df = pd.read_csv('Titanic-Dataset.csv')
print("Initial Info:")
print(df.info())

imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']])

df = df.drop('Cabin', axis=1)

print("\nStep 2: After handling missing values and dropping 'Cabin'\n")
print(df.info())

df = df.drop(['Name', 'Ticket'], axis=1)

categorical_features = ['Sex', 'Embarked']
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])
processed_data = preprocessor.fit_transform(df)
print("\nStep 3: After encoding categorical features and scaling numerical ones\n")
print("Shape of the processed data:", processed_data.shape)

print("\nStep 4: Visualizing Outliers (for demonstration)\n")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.title('Boxplot of Numerical Features to Detect Outliers')
plt.show()

print("\nData preprocessing complete. The processed data is ready for model training.")
