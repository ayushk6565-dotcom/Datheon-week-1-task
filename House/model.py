import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Dataset
df = pd.read_csv("House/housing.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# 2. Feature Engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

df["income_house_age"] = df["median_income"] * df["housing_median_age"]

# 3. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 4. Separate Features and Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Identify Columns
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns

# 7. Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 8. Linear Regression (OLS)
ols = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

ols.fit(X_train, y_train)
ols_pred = ols.predict(X_test)

# 9. Ridge Regression with GridSearchCV
ridge = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge())
])

ridge_params = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

ridge_grid = GridSearchCV(
    ridge, ridge_params, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

ridge_grid.fit(X_train, y_train)
ridge_pred = ridge_grid.predict(X_test)

# 10. Lasso Regression with GridSearchCV
lasso = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Lasso(max_iter=20000))
])

lasso_params = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10]
}

lasso_grid = GridSearchCV(
    lasso, lasso_params, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

lasso_grid.fit(X_train, y_train)
lasso_pred = lasso_grid.predict(X_test)

# 11. Model Evaluation
models = {
    "OLS": ols_pred,
    "Ridge": ridge_pred,
    "Lasso": lasso_pred
}

results = []

for name, pred in models.items():

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    results.append([name, rmse, mae, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2 Score"]
)

print("\nModel Comparison:")
print(results_df)

print("\nBest Ridge Alpha:", ridge_grid.best_params_)
print("Best Lasso Alpha:", lasso_grid.best_params_)

# 12. Predicted vs Actual Plot
plt.figure(figsize=(8, 6))

for name, pred in models.items():
    plt.scatter(y_test, pred, alpha=0.3, label=name)

plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()

# 13. Residual Distribution
plt.figure(figsize=(8, 6))

for name, pred in models.items():
    residuals = y_test - pred
    sns.kdeplot(residuals, label=name)

plt.xlabel("Residuals")
plt.title("Residual Distribution")
plt.legend()
plt.show()
