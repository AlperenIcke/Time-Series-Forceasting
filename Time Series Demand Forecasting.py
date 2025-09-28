import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Task 1: Dataset Selection and Exploration

# Load Data
df = pd.read_csv(r"C:\Users\Alperen\OneDrive\Masaüstü\Academic Files\Term1\Fundamentals of Data Analytics\Time Series Forecasting - Assignment\PRICE_AND_DEMAND_201806_NSW1.csv")

# Initial Data Inspection
print("\nDataset Head:")
print(df.head())

# Data Cleaning and Basic Preprocessing
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
df = df.set_index('SETTLEMENTDATE')
df = df.sort_index()

target_col = 'TOTALDEMAND'
exog_col = 'RRP'

if target_col in df.columns and df[target_col].isnull().any():
    df[target_col] = df[target_col].ffill()
if exog_col in df.columns and df[exog_col].isnull().any():
    df[exog_col] = df[exog_col].ffill()

# Drop rows where target is still have missing values
df.dropna(subset=[target_col], inplace=True)

# Plot of the target variable (TOTALDEMAND) over time
plt.figure(figsize=(12, 6))
df[target_col].plot()
plt.title(f'{target_col} Over Time')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.tight_layout()
plt.show()

# Distribution of TOTALDEMAND
plt.figure(figsize=(8, 5))
sns.histplot(df[target_col], kde=True, bins=30)
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Task 2: Data Preprocessing and Feature Engineering
df_feat = df.copy()

# 2.1 Create Time-Based Features
df_feat['hour'] = df_feat.index.hour
df_feat['dayofweek'] = df_feat.index.dayofweek # Monday=0, Sunday=6

# 2.2 Create Lagged Features
# Lag by 1 period, 2 periods, 1 day (30-min data)
lags_demand = [1, 2, 48]
if target_col in df_feat.columns:
    for lag in lags_demand:
        if len(df_feat) > lag:
            df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)

# Handle NaNs created by lags
original_len = len(df_feat)
df_feat = df_feat.dropna()


# 2.4 Feature Selection and Scaling
potential_X_df = df_feat.drop(columns=[target_col], errors='ignore')
X = potential_X_df.select_dtypes(include=np.number)
y = df_feat[target_col]
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

# Split data: 70% train, 15% validation, 15% test (chronological)
train_size_frac = 0.7
val_size_frac = 0.15
test_size_frac = 0.15

n = len(X)
train_end_idx = int(n * train_size_frac)
val_end_idx = int(n * (train_size_frac + val_size_frac))

X_train, y_train = X.iloc[:train_end_idx], y.iloc[:train_end_idx]
X_val, y_val = X.iloc[train_end_idx:val_end_idx], y.iloc[train_end_idx:val_end_idx]
X_test, y_test = X.iloc[val_end_idx:], y.iloc[val_end_idx:]

# Normalize/Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Task 3: Regression Model Development and Evaluation

# Evaluation
def evaluate_model(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

model_results = {}

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions on validation set (Linear Regression)
y_pred_lr_val = lr_model.predict(X_val_scaled)
model_results['Linear Regression (Validation)'] = evaluate_model(y_val, y_pred_lr_val, "Linear Regression (Validation)")

# Feature importance (coefficients)
if hasattr(lr_model, 'coef_') and X_train.shape[1] > 0 :
    lr_coeffs = pd.DataFrame({'feature': X_train.columns, 'coefficient': lr_model.coef_})
    lr_coeffs['abs_coefficient'] = lr_coeffs['coefficient'].abs()
    lr_coeffs = lr_coeffs.sort_values('abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])
    print("\nLinear Regression - Top Feature Coefficients (Absolute Value):")
    print(lr_coeffs.head(min(5, len(lr_coeffs))))

# Model 2: Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
rf_model.fit(X_train_scaled, y_train)

# Predictions on validation set (Random Forest Regression)
y_pred_rf_val = rf_model.predict(X_val_scaled)
model_results['Random Forest (Validation)'] = evaluate_model(y_val, y_pred_rf_val, "Random Forest (Validation)")

# Feature importance
if hasattr(rf_model, 'feature_importances_') and X_train.shape[1] > 0:
    rf_importances = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
    rf_importances = rf_importances.sort_values('importance', ascending=False)
    print("\nRandom Forest - Top Feature Importances:")
    print(rf_importances.head(min(5, len(rf_importances))))

# 3.3 Final Evaluation on Test Set

# Linear Regression on Test Set
y_pred_lr_test = lr_model.predict(X_test_scaled)
model_results['Linear Regression (Test)'] = evaluate_model(y_test, y_pred_lr_test, "Linear Regression (Test)")

# Random Forest on Test Set
y_pred_rf_test = rf_model.predict(X_test_scaled)
model_results['Random Forest (Test)'] = evaluate_model(y_test, y_pred_rf_test, "Random Forest (Test)")

best_model_r2_test = max(model_results['Linear Regression (Test)']['R2'], model_results['Random Forest (Test)']['R2'])
best_model_name = "Random Forest" if model_results['Random Forest (Test)']['R2'] >= model_results['Linear Regression (Test)']['R2'] else "Linear Regression"
y_pred_best_test = y_pred_rf_test if best_model_name == "Random Forest" else y_pred_lr_test

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label=f'Actual {target_col}', alpha=0.8)
plt.plot(y_test.index, y_pred_best_test, label=f'Predicted {target_col} ({best_model_name})', alpha=0.8, linestyle='--')
plt.title(f'Actual vs. Predicted {target_col} on Test Set ({best_model_name})')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.tight_layout()
plt.show()