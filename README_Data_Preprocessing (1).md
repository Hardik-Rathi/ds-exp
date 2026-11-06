# Data Preprocessing: Skewness, Scaling and Encoding

This repository contains the Python script and Jupyter Notebook for preprocessing the **Sub-Division IMD 2017 rainfall dataset**.

## Dataset
The CSV contains yearly rainfall measurements for Indian subdivisions.
- **Nominal categorical feature:** `SUBDIVISION`
- **Numerical features:** `YEAR`, monthly rainfall (`JAN`–`DEC`), annual rainfall (`ANNUAL`) and seasonal aggregates (`JF`, `MAM`, `JJAS`, `OND`).

There is no naturally ordinal categorical column in the source CSV. To demonstrate ordinal encoding transparently, the notebook derives `YEAR_PERIOD` from `YEAR`, with the ordered categories `Early`, `Middle`, and `Recent`.

## Implemented tasks
1. Load and inspect the dataset, data types and missing values.
2. Fill missing numerical values with column medians and categorical values with the mode.
3. Study skewness using histograms and KDE plots for `JAN`, `JUL`, and `ANNUAL`.
4. Implement manual Min-Max scaling using `(x - min) / (max - min)`.
5. Apply Scikit-Learn `MinMaxScaler` and compare both methods.
6. Demonstrate ordinal encoding of the derived `YEAR_PERIOD`.
7. One-hot encode `SUBDIVISION` using `pandas.get_dummies(drop_first=True)`.
8. Compare original and transformed datasets.
9. Verify normalized minimum/maximum and standardized mean/population standard deviation.

## Files
- `Experiment_Data_Preprocessing.ipynb` — notebook with explanations, code and plots.
- `Experiment_Data_Preprocessing.py` — raw Python code.
- `Sub_Division_IMD_2017(2).csv` — original dataset.

## Requirements and installation
Python 3.x, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run
Keep the CSV in the same folder as the notebook/script.

```bash
python Experiment_Data_Preprocessing.py
```

Or open `Experiment_Data_Preprocessing.ipynb` in Jupyter Notebook / Google Colab and run all cells.

## Notes on interpretation
- `YEAR` is retained as a time identifier and is not Min-Max scaled.
- Min-Max scaling is applied to rainfall measurement columns.
- `StandardScaler` is additionally applied to verify mean approximately 0 and population standard deviation approximately 1. This is separate from Min-Max normalization.
- `drop_first=True` omits one subdivision reference column to avoid redundant dummy variables when a model includes an intercept.
