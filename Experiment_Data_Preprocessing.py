# Experiment: Skewness, Min-Max Scaling and Encoding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Make display() work in both notebooks and ordinary Python environments.
try:
    from IPython.display import display
except ImportError:
    display = print

# 1. Load the dataset
df = pd.read_csv("Sub_Division_IMD_2017(2).csv")
print("Shape:", df.shape)
display(df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
print("\nNumerical features:", numeric_cols)
print("Categorical features:", categorical_cols)

# Fill missing values in a working copy (median for numeric, mode for categorical).
data = df.copy()
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# 2. Skewness with histogram and KDE
selected = ["JAN", "JUL", "ANNUAL"]
print("\nSkewness:\n", data[selected].skew())
for col in selected:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Histogram and KDE: {col}")
    plt.xlabel("Rainfall")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# 3. Manual Min-Max scaling: X_scaled = (X - X_min) / (X_max - X_min)
# YEAR is retained as a time identifier; rainfall measurements are scaled.
scale_cols = [c for c in numeric_cols if c != "YEAR"]
manual_scaled = data[scale_cols].copy()
for col in scale_cols:
    xmin, xmax = data[col].min(), data[col].max()
    manual_scaled[col] = 0.0 if xmax == xmin else (data[col] - xmin) / (xmax - xmin)

# Scikit-Learn MinMaxScaler
scaler = MinMaxScaler()
sklearn_scaled = pd.DataFrame(
    scaler.fit_transform(data[scale_cols]),
    columns=scale_cols, index=data.index
)
print("\nManual Min-Max results:")
display(manual_scaled.head())
print("\nScikit-Learn MinMaxScaler results:")
display(sklearn_scaled.head())
print("Maximum difference between manual and Scikit-Learn:",
      np.abs(manual_scaled.to_numpy() - sklearn_scaled.to_numpy()).max())

# 4. Ordinal label encoding demonstration.
# The CSV has no naturally ordinal categorical field. Derive an explicitly
# documented chronological category from YEAR instead of mislabeling SUBDIVISION.
data["YEAR_PERIOD"] = pd.cut(
    data["YEAR"], bins=[1900, 1940, 1980, 2020],
    labels=["Early", "Middle", "Recent"], include_lowest=True
)
ordinal_mapping = {"Early": 0, "Middle": 1, "Recent": 2}
data["YEAR_PERIOD_ENCODED"] = data["YEAR_PERIOD"].map(ordinal_mapping).astype(int)
print("\nOrdinal mapping:", ordinal_mapping)
display(data[["YEAR", "YEAR_PERIOD", "YEAR_PERIOD_ENCODED"]].head())

# 5. One-hot encode nominal SUBDIVISION; drop_first prevents redundant dummy column.
one_hot = pd.get_dummies(data["SUBDIVISION"], prefix="SUBDIVISION",
                         drop_first=True, dtype=int)
print("\nOne-hot encoded features:", one_hot.shape[1])
display(one_hot.head())

# 6. Original vs transformed data
transformed = pd.concat([
    data[["YEAR"]].reset_index(drop=True),
    manual_scaled.add_suffix("_MINMAX").reset_index(drop=True),
    data[["YEAR_PERIOD_ENCODED"]].reset_index(drop=True),
    one_hot.reset_index(drop=True)
], axis=1)
print("\nOriginal dataset:")
display(df.head())
print("\nTransformed dataset:")
display(transformed.head())

# Verify normalized min=0 and max=1
minmax_check = pd.DataFrame({
    "minimum": sklearn_scaled.min(),
    "maximum": sklearn_scaled.max()
})
print("\nMin-Max verification (expected 0 and 1):")
display(minmax_check)

# Standardization is also shown because the requested verification mentions mean=0, std=1.
# StandardScaler uses population standard deviation (ddof=0).
standardized = pd.DataFrame(
    StandardScaler().fit_transform(data[scale_cols]), columns=scale_cols
)
standard_check = pd.DataFrame({
    "mean": standardized.mean(),
    "population_std": standardized.std(ddof=0)
})
print("\nStandardization verification (expected mean≈0, std≈1):")
display(standard_check)

print("\nInterpretation:")
print("Positive skewness means a longer right tail; negative skewness means a longer left tail.")
print("Min-Max normalization places rainfall features between 0 and 1.")
print("Standardization centers features near 0 with population standard deviation 1.")
print("YEAR_PERIOD is a derived ordinal variable; SUBDIVISION is nominal.")
