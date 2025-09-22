# Elevate Labs - Task 1: Data Cleaning & Preprocessing

### **Objective**
The goal of this task was to perform essential data cleaning and preprocessing steps on a raw dataset to prepare it for a machine learning model. The process involved handling missing values, encoding categorical variables, and scaling numerical features.

### **Dataset**
The project utilized the Titanic dataset, which contains information about passengers aboard the Titanic.

### **Steps Taken**

1.  **Data Exploration**: I started by loading the dataset and examining its structure using `df.info()`. This step revealed missing values in the `Age`, `Cabin`, and `Embarked` columns.

2.  **Handling Missing Values**:
    * The `Age` column's missing values were imputed with the median to maintain the data's distribution.
    * The `Embarked` column's missing values were filled with the mode (the most frequent value).
    * The `Cabin` column, which had a significant number of missing values, was dropped from the dataset.

3.  **Feature Engineering & Encoding**:
    * Irrelevant columns such as `Name` and `Ticket` were dropped.
    * Categorical features like `Sex` and `Embarked` were converted into a numerical format using **One-Hot Encoding**.

4.  **Feature Scaling**:
    * Numerical features (`Age`, `Fare`, `SibSp`, `Parch`) were scaled using **StandardScaler** to ensure they have a similar scale, which is crucial for many machine learning algorithms.

### **Conclusion**
The dataset has been successfully cleaned and preprocessed, resulting in a numerical array ready to be fed into a machine learning model for training and prediction.
