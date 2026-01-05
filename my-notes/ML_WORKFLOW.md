# ML Engineer Data Workflow - Quick Reference Guide

## Overview

This guide documents the typical data workflow for machine learning projects, covering data structures, conversions, and best practices learned from the ML-For-Beginners curriculum.

---

## Data Structures in ML Projects

### When to Use Each Structure

| Task | Data Structure | Why |
|------|---------------|-----|
| **Data loading** | Pandas DataFrame | Handles CSV, Excel, JSON, SQL easily |
| **Data cleaning** | Pandas DataFrame | Rich methods for nulls, filtering, strings |
| **Exploration/EDA** | Pandas DataFrame | `.describe()`, `.groupby()`, `.corr()` |
| **Visualization** | Pandas DataFrame | Integrated plotting, column names |
| **Model training (X)** | NumPy Array (2D) | sklearn requirement |
| **Model training (y)** | Pandas Series or NumPy | sklearn accepts both |
| **Predictions** | NumPy Array | Model outputs NumPy arrays |
| **Results analysis** | Pandas DataFrame | Easy comparison and visualization |

---

## Complete ML Workflow

### 1. Data Acquisition

```python
import pandas as pd
import numpy as np

# CSV files (most common)
df = pd.read_csv('data.csv')

# Excel files
df = pd.read_excel('data.xlsx')

# JSON files (APIs, web data)
df = pd.read_json('data.json')

# SQL databases
import sqlalchemy
engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/db')
df = pd.read_sql('SELECT * FROM table', engine)

# Parquet files (big data)
df = pd.read_parquet('data.parquet')

# sklearn datasets (already NumPy arrays)
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)
```

### 2. Data Cleaning & Exploration (Pandas)

```python
# Explore the data
print(df.info())           # Column types, null counts
print(df.describe())       # Statistical summary
print(df.isnull().sum())   # Count nulls per column
print(df.head())           # First 5 rows

# Clean the data
df.dropna(inplace=True)                    # Remove missing values
df = df[df['column'] > threshold]          # Filter rows
df = df[df['Package'].str.contains('bushel')]  # String filtering
df.drop_duplicates(inplace=True)           # Remove duplicates

# Feature engineering
df['Price'] = (df['Low Price'] + df['High Price']) / 2
df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear
df['Month'] = pd.to_datetime(df['Date']).dt.month

# Handle outliers
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= Q1 - 1.5*IQR) & (df['Price'] <= Q3 + 1.5*IQR)]

# Select relevant columns
columns_to_select = ['Feature1', 'Feature2', 'Target']
df = df.loc[:, columns_to_select]
```

### 3. Exploratory Data Analysis (Pandas + Matplotlib)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
df.describe()
df.corr()

# Pandas built-in plotting
df['Price'].hist(bins=30)
df.plot.scatter('DayOfYear', 'Price')
df.groupby('Variety')['Price'].mean().plot(kind='bar')

# Matplotlib for custom plots
plt.figure(figsize=(10, 6))
plt.scatter(df['Feature'], df['Target'], alpha=0.5)
plt.xlabel('Feature Name')
plt.ylabel('Target Variable')
plt.title('Relationship Between Feature and Target')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### 4. Prepare Data for Model (Convert to NumPy)

```python
from sklearn.model_selection import train_test_split

# IMPORTANT: X must be 2D NumPy array for sklearn

# Single feature (must reshape to 2D)
X = df['Feature'].to_numpy().reshape(-1, 1)

# Multiple features (already 2D)
X = df[['Feature1', 'Feature2', 'Feature3']].to_numpy()

# Target variable (can stay as Series or convert)
y = df['Target'].to_numpy()  # or just df['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
```

### 5. Train Model (NumPy Arrays)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions (returns NumPy array)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print(f"Train R² Score: {train_r2:.3f}")
print(f"Test R² Score: {test_r2:.3f}")
print(f"Test RMSE: {test_rmse:.2f}")
```

### 6. Visualize Results (Convert Back to DataFrame)

```python
# Create results DataFrame for easy analysis
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Residual': y_test - y_test_pred
})

# If using multiple features, include them
if X_test.shape[1] > 1:
    for i in range(X_test.shape[1]):
        results_df[f'Feature_{i}'] = X_test[:, i]
else:
    results_df['Feature'] = X_test.flatten()

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Predicted'], results_df['Residual'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.show()

# Plot 3: Feature vs Actual/Predicted
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Feature'], results_df['Actual'], 
            label='Actual', alpha=0.5, color='black')
plt.scatter(results_df['Feature'], results_df['Predicted'], 
            label='Predicted', alpha=0.5, color='blue')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Feature vs Target (Actual and Predicted)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 7. Save and Deploy Model

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved to 'model.pkl'")

# Load model for production use
loaded_model = joblib.load('model.pkl')

# Make prediction with new data
new_data = np.array([[250]])  # Single feature example
prediction = loaded_model.predict(new_data)
print(f"Prediction: {prediction[0]:.2f}")

# Flask API example
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['feature']]])
    prediction = loaded_model.predict(features)
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Evaluate Model 
![alt text](image.png)

## Key Conversion Patterns

### Pandas → NumPy

```python
# Single feature (MUST reshape to 2D)
X = df['feature'].to_numpy().reshape(-1, 1)

# Multiple features (already 2D)
X = df[['feat1', 'feat2']].to_numpy()

# Target variable
y = df['target'].to_numpy()  # or just df['target']
```

### NumPy → Pandas

```python
# Create DataFrame from NumPy array
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# Add predictions to existing DataFrame
df['Predictions'] = model.predict(X)

# Create results DataFrame
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
```

---

## Common Data Cleaning Operations

### Handling Missing Values

```python
# Check for missing values
df.isnull().sum()
df.info()

# Remove rows with any missing values
df.dropna(inplace=True)

# Remove rows with missing values in specific columns
df.dropna(subset=['column1', 'column2'], inplace=True)

# Fill missing values
df.fillna(0, inplace=True)  # Fill with 0
df.fillna(df.mean(), inplace=True)  # Fill with mean
df['column'].fillna(df['column'].median(), inplace=True)  # Fill with median
```

### Filtering Data

```python
# Numeric filtering
df = df[df['Price'] > 0]
df = df[(df['Price'] >= 10) & (df['Price'] <= 100)]

# String filtering
df = df[df['Category'] == 'A']
df = df[df['Name'].str.contains('text')]
df = df[df['City'].isin(['Boston', 'Chicago'])]

# Date filtering
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2020-01-01']
```

### Handling Duplicates

```python
# Check for duplicates
df.duplicated().sum()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove duplicates based on specific columns
df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
```

---

## Understanding NumPy Array Indexing

### Visual Representation

```
NumPy 2D Array (Matrix):
    Column:  0      1      2      3
           [age]  [sex]  [bmi]  [bp]
Row 0    [ 0.03,  0.05, -0.04,  0.02 ]
Row 1    [-0.01, -0.04,  0.05,  0.03 ]
Row 2    [ 0.08,  0.05,  0.00, -0.03 ]
         ↑                      ↑
         |                      |
     X[:, 0]                X[:, 3]
    (all rows,             (all rows,
   column 0)              column 3)
```

### Indexing Syntax

```python
# X[row_selector, column_selector]

X[0, 0]      # Single value: row 0, column 0
X[0, :]      # Entire row: row 0, all columns (1D array)
X[:, 3]      # Entire column: all rows, column 3 (1D array)
X[0:5, 3]    # Rows 0-4, column 3 (1D array)
X[:, [0,2]]  # All rows, columns 0 and 2 (2D array)

# The colon : means "all" in that dimension
```

---

## Pandas vs NumPy Features

### What Pandas Provides

```python
# Named columns (readability)
df['Price']  # Clear what this represents
vs
X[:, 3]      # What is column 3? Need to remember

# Data cleaning methods
df.dropna()
df.isnull().sum()
df.duplicated()
vs
# NumPy requires manual mask creation

# String operations
df[df['Package'].str.contains('bushel')]
vs
# NumPy doesn't handle strings well

# GroupBy and aggregation
df.groupby('Category')['Value'].mean()
vs
# NumPy requires manual loops

# DateTime handling
pd.to_datetime(df['Date']).dt.dayofyear
vs
# NumPy has limited datetime support

# Integrated plotting
df.plot.scatter('x', 'y')
vs
plt.scatter(X[:, 0], X[:, 1])  # Must extract columns manually
```

---

## Common Mistakes to Avoid

### 1. Forgetting to Reshape Single Features

```python
# ❌ WRONG - 1D array
X = df['feature'].to_numpy()
# Shape: (100,)

# ✅ CORRECT - 2D array
X = df['feature'].to_numpy().reshape(-1, 1)
# Shape: (100, 1)
```
Memory Aid: The Feature Matrix Rule
sklearn's Feature Matrix (X) must always be 2D:

Number of dimensions | Shape format    | When to use
---------------------|----------------|------------------
1D                   | (n,)           | Never for sklearn X
2D                   | (n, 1)         | Single feature
2D                   | (n, m)         | Multiple features (m > 1)

n = number of samples
m = number of features

Quick Decision Tree:
Do I have a single feature?
├─ Yes → Use .reshape(-1, 1)
│        X = df['feature'].to_numpy().reshape(-1, 1)
│
└─ No (multiple features) → No reshape needed
         X = df[['feat1', 'feat2']].to_numpy()

Key Takeaway: Always check your X shape before training. If it's 1D ((n,)), reshape it to 2D ((n, 1)) using .reshape(-1, 1).

```
# Target variable can be 1D or 2D - both work!

# ✅ 1D target (most common)
y = df['Price'].to_numpy()
print(y.shape)  # (100,)

# ✅ 2D target (also works)
y = df['Price'].to_numpy().reshape(-1, 1)
print(y.shape)  # (100, 1)

# sklearn accepts both because:
# - Target is always a single variable (not multiple columns)
# - Clear interpretation: each value is the target for one sample
```

### 2. Running Cells Multiple Times

```python
# ❌ This fails on second run
X = X[:, 2]  # First run: X is now 1D
X = X[:, 2]  # Second run: Error! Can't index 1D array

# ✅ Always reload data first
X, y = datasets.load_diabetes(return_X_y=True)
X = X[:, 2]
```

### 3. Using NumPy for Data Cleaning

```python
# ❌ HARD with NumPy
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]

# ✅ EASY with Pandas
df.dropna(inplace=True)
```

### 4. Mixing Data Types

```python
# ❌ Inconsistent
X = df['feature'].to_numpy().reshape(-1, 1)
y = df['target']  # Still a Pandas Series

# ✅ Consistent (both NumPy)
X = df['feature'].to_numpy().reshape(-1, 1)
y = df['target'].to_numpy()
```

---

## sklearn Datasets vs Real-World Data

### sklearn Datasets (e.g., load_diabetes)

```python
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)

# Already:
# - NumPy arrays ✓
# - Pre-cleaned ✓
# - Normalized ✓
# - No missing values ✓
# - Ready for training ✓

# No Pandas needed - go straight to modeling
```

### Real-World CSV Data

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Needs:
# - Loading with Pandas
# - Cleaning (nulls, duplicates)
# - Filtering
# - Feature engineering
# - Type conversions
# - Then convert to NumPy for training

# Pandas essential for this workflow
```

---

## Quick Reference Cheatsheet

### Data Pipeline Flow

```
CSV/SQL/API (Raw Data)
    ↓
Pandas DataFrame (Cleaning & EDA)
    ↓
NumPy Arrays (Model Training)
    ↓
NumPy Arrays (Predictions)
    ↓
Pandas DataFrame (Results Visualization)
    ↓
JSON/Pickle/Database (Deployment)
```

### Essential Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
```

### Key Methods to Remember

**Pandas:**
- `.info()`, `.describe()`, `.head()`
- `.dropna()`, `.isnull()`, `.fillna()`
- `.groupby()`, `.agg()`, `.corr()`
- `.to_numpy()`, `.plot()`

**NumPy:**
- `.shape`, `.dtype`, `.reshape()`
- `.mean()`, `.std()`, `.min()`, `.max()`
- Array indexing: `X[:, col]`, `X[row, :]`

**sklearn:**
- `train_test_split(X, y, test_size, random_state)`
- `model.fit(X_train, y_train)`
- `model.predict(X_test)`
- `mean_squared_error()`, `r2_score()`

---

## Lessons Learned

### From Lesson 2-1: Regression Tools
- sklearn datasets come as NumPy arrays
- `.reshape(-1, 1)` converts 1D to 2D for single features
- NumPy array indexing: `X[:, 3]` = all rows, column 3
- Colon `:` means "all" in that dimension

### From Lesson 2-2: Data Preparation
- Use Pandas for data cleaning and manipulation
- `.groupby()` for powerful aggregations
- Pandas `.plot()` for quick visualizations
- Filter DataFrames easily with boolean indexing

### From Lesson 2-3: Linear Regression
- Convert Pandas to NumPy before training
- sklearn accepts Pandas Series for y
- Convert predictions back to DataFrame for analysis
- Always split data before training

---

## Additional Resources

- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **NumPy Documentation**: https://numpy.org/doc/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **Seaborn Tutorial**: https://seaborn.pydata.org/tutorial.html
