# Ames House Prices — Regression & Ensemble

A machine learning project based on the **Kaggle House Prices: Advanced Regression Techniques** competition.

The goal is to predict residential house sale prices using the Ames Housing dataset. The project focuses on building a complete regression pipeline, including exploratory data analysis, feature engineering, preprocessing, model comparison, hyperparameter tuning, and ensemble learning.

## Kaggle Competition

**Competition:** House Prices: Advanced Regression Techniques

The competition evaluates predictions using **Root Mean Squared Logarithmic Error (RMSLE)**.

The target variable is `SalePrice`, which is log-transformed during training to better handle its right-skewed distribution.

## Project Workflow

### 1. Exploratory Data Analysis

* Examined numerical and categorical features
* Analyzed missing values and their meanings
* Investigated feature distributions and skewness
* Studied correlations between features and `SalePrice`
* Identified highly correlated feature pairs
* Investigated potential outliers

### 2. Data Preprocessing

* Removed the `Id` column
* Converted `MSSubClass` to categorical
* Handled missing values according to their semantic meaning
* Used `0` for features where missing values represent the absence of a property component
* Applied ordinal encoding to ordered categorical features
* Applied one-hot encoding to nominal categorical features
* Applied median imputation where appropriate
* Created a custom `LotFrontage` imputation strategy using neighborhood-level medians
* Applied `log1p` transformation to skewed numerical features
* Standardized numerical features using `StandardScaler`

### 3. Target Transformation

Because `SalePrice` is strongly right-skewed, the target was transformed using:

```python
y = np.log1p(SalePrice)
```

Predictions were converted back to the original price scale using:

```python
np.expm1(predictions)
```

### 4. Outlier Handling

Two extreme `GrLivArea` observations were removed because they represented unusually large living areas with disproportionately low sale prices.

## Models

Several regression algorithms were compared using 5-fold cross-validation:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Support Vector Regression (SVR)
* K-Nearest Neighbors Regression
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression

## Hyperparameter Tuning

Hyperparameter optimization was performed using:

* `GridSearchCV` for Lasso and Ridge
* `RandomizedSearchCV` for Gradient Boosting and XGBoost

The tuned models were then compared using cross-validation RMSE.

## Ensemble Model

The final approach combined the strongest individual models using a `VotingRegressor` consisting of:

* Lasso
* Gradient Boosting
* XGBoost

The ensemble achieved the best cross-validation result:

**RMSE: 0.1077 ± 0.0083**

This outperformed the individual tuned models.

## Results

| Model               | Cross-Validation RMSE |
| ------------------- | --------------------: |
| Lasso               |              0.113864 |
| Ridge               |              0.115811 |
| Gradient Boosting   |              0.115904 |
| XGBoost             |              0.116697 |
| SVR                 |              0.130130 |
| Linear Regression   |              0.132397 |
| Random Forest       |              0.138040 |
| KNN                 |              0.170112 |
| **Voting Ensemble** |            **0.1077** |

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

## Repository Structure

```text
Ames-House-Prices-Clean-Pipeline-Ensemble-LB-0.123-/
│
├── house_price.ipynb
└── README.md
```

## Dataset

The project uses the **Ames Housing dataset provided through the Kaggle competition**.

The original Kaggle competition files are not included in this repository. The notebook was developed and executed using the Kaggle dataset environment.

## Competition

This project was developed as part of the Kaggle **House Prices: Advanced Regression Techniques** competition and focuses on building a robust end-to-end regression and ensemble pipeline.
