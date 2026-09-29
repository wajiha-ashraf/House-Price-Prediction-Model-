# Karachi House Price Prediction

A machine learning regression project that predicts house prices in Karachi using property-related features.

## Dataset

The notebook uses `karachi_house_prices.csv`.

The dataset contains **2,000 rows** and the model uses **8 input features** to predict `Price_PKR`.

### Features

| Feature | Role |
|---|---|
| `Location` | Input feature |
| `Condition` | Input feature |
| `Parking` | Input feature |
| `Year_Built` | Input feature |
| `Floors` | Input feature |
| `Bathrooms` | Input feature |
| `Bedrooms` | Input feature |
| `Size_Marla` | Input feature |
| `Price_PKR` | Target |

## Data Preparation

The notebook performs the following steps:

1. Loads the Karachi house-price dataset.
2. Inspects the dataset using `head()`, `tail()`, `shape`, `dtypes`, `describe()`, and column checks.
3. Checks for missing values.
4. Encodes `Location`, `Condition`, and `Parking` using `LabelEncoder`.
5. Separates the features from the target:
   - `X` = 8 input features
   - `y` = `Price_PKR`

## Train-Test Split

The dataset is divided using:

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

The resulting split is:

- Training set: **1,600 samples**
- Test set: **400 samples**

## Feature Scaling

`StandardScaler` is fitted on the training features and then applied to both the training and test sets.

## Model

The project uses a **Random Forest Regressor**:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The model is trained on the training data and used to predict house prices for the test set.

## Results

The notebook evaluates the model using R² and RMSE.

| Metric | Result |
|---|---:|
| **R² Score** | **0.97097** |
| **RMSE** | **3,193,714.32 PKR** |

The R² score indicates that the model explains a high proportion of the variation in the test-set house prices according to this evaluation.

## Correlation Analysis

The notebook calculates correlations with `Price_PKR`.

| Feature | Correlation with Price_PKR |
|---|---:|
| `Location` | -0.390027 |
| `Condition` | -0.012693 |
| `Parking` | 0.003758 |
| `Year_Built` | 0.042076 |
| `Floors` | 0.047003 |
| `Bathrooms` | 0.048938 |
| `Bedrooms` | 0.068730 |
| `Size_Marla` | 0.633544 |

Among the listed features, `Size_Marla` has the strongest positive correlation with `Price_PKR`.

## Feature Importance

The Random Forest model reports the following feature-importance values:

| Feature | Importance |
|---|---:|
| `Location` | 0.485170 |
| `Size_Marla` | 0.434920 |
| `Condition` | 0.056314 |
| `Year_Built` | 0.010748 |
| `Bedrooms` | 0.004195 |
| `Bathrooms` | 0.003939 |
| `Floors` | 0.003313 |
| `Parking` | 0.001401 |

The model's feature-importance output gives the highest importance to `Location` and `Size_Marla`.

## Visualizations

The notebook includes:

1. Feature importance bar chart
2. Actual vs. predicted house-price scatter plot
3. Residual plot

The actual-vs-predicted plot compares the model's predicted prices with the actual test-set prices.

The residual plot shows the difference between actual and predicted prices:

```text
Residual = Actual Price - Predicted Price
```

## Project Workflow

```text
Karachi House Price Dataset
          ↓
Data Inspection
          ↓
Missing Value Check
          ↓
Categorical Encoding
          ↓
Feature / Target Separation
          ↓
Train-Test Split
          ↓
Standard Scaling
          ↓
Random Forest Regressor
          ↓
Price Prediction
          ↓
R² + RMSE Evaluation
          ↓
Feature Importance / Residual Analysis
```

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Main Scikit-learn components:

- `LabelEncoder`
- `StandardScaler`
- `train_test_split`
- `RandomForestRegressor`
- `mean_squared_error`
- `r2_score`

## Important Notebook Note

The notebook contains a second encoding section near the end where `Location`, `Parking`, and `Condition` are encoded again after they have already been converted to numeric values.

The first encoding step is the relevant preprocessing step for the model. The later re-encoding section appears to be exploratory/checking code rather than a necessary part of the training pipeline.

Also, `StandardScaler` is used before the Random Forest model. Scaling is generally not required for Random Forest, but it is present in the notebook's implementation.

These points are documented here rather than silently changing the notebook's implementation.

## How to Run

1. Place `karachi_house_prices.csv` in the project directory.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Update the dataset path in `pd.read_csv()` if necessary.
4. Run the notebook cells from top to bottom.

## Project Status

The notebook contains a complete training and evaluation workflow for the house-price regression model, including preprocessing, model training, predictions, evaluation metrics, feature importance, and residual analysis.
