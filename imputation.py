import pandas as pd
import numpy as np

complete_data = pd.read_excel("complete_dataset.xlsx")
incomplete_data = pd.read_excel("incomplete_dataset.xlsx")

complete_data['source'] = 'complete'
incomplete_data['source'] = 'incomplete'

missing_cols = [col for col in complete_data.columns if col not in incomplete_data.columns and col != 'source']
for col in missing_cols:
    incomplete_data[col] = np.nan

ordered_features = complete_data.drop(columns=['source']).columns.tolist()
incomplete_data = incomplete_data[ordered_features + ['source']]

combined_data = pd.concat([complete_data, incomplete_data], ignore_index=True)

numeric_cols = combined_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = combined_data.select_dtypes(include=['object']).columns.tolist()
if 'source' in numeric_cols: numeric_cols.remove('source')
if 'source' in categorical_cols: categorical_cols.remove('source')

numeric_data = combined_data[numeric_cols]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

use_rf = True  #True to use RandomForestRegressor; False for BayesianRidge.
if use_rf:
    from sklearn.ensemble import RandomForestRegressor
    num_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    from sklearn.linear_model import BayesianRidge
    num_estimator = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

iterative_imputer = IterativeImputer(
    estimator=num_estimator,
    max_iter=10,
    random_state=42,
    initial_strategy='mean'
)
imputed_scaled = iterative_imputer.fit_transform(numeric_data_scaled)
imputed_numeric = scaler.inverse_transform(imputed_scaled)
numeric_imputed_df = pd.DataFrame(imputed_numeric, columns=numeric_cols)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

def impute_categorical_rf(full_data, cat_cols, random_state=42, n_estimators=100):
    data_imputed = full_data.copy()
    for col in cat_cols:
        missing_mask = data_imputed[col].isnull()
        if missing_mask.sum() == 0:
            continue
        predictors = [c for c in data_imputed.columns if c not in [col, 'source']]
        X = data_imputed[predictors].copy()
        cat_predictors = X.select_dtypes(include=['object']).columns.tolist()
        for cp in cat_predictors:
            X[cp] = X[cp].fillna("missing").astype(str)
        encoder = OrdinalEncoder()
        X[cat_predictors] = encoder.fit_transform(X[cat_predictors])
        X_train = X.loc[~missing_mask]
        X_test = X.loc[missing_mask]
        y_train = data_imputed.loc[~missing_mask, col].fillna("missing").astype(str).values.reshape(-1, 1)
        target_enc = OrdinalEncoder()
        y_train_enc = target_enc.fit_transform(y_train).ravel()
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train_enc)
        y_pred_enc = clf.predict(X_test)
        y_pred = target_enc.inverse_transform(y_pred_enc.reshape(-1, 1)).ravel()
        data_imputed.loc[missing_mask, col] = y_pred
    return data_imputed[cat_cols]

categorical_imputed_df = impute_categorical_rf(combined_data, categorical_cols, random_state=42, n_estimators=100)

imputed_combined = pd.concat([
    numeric_imputed_df.reset_index(drop=True),
    categorical_imputed_df.reset_index(drop=True),
    combined_data[['source']].reset_index(drop=True)
], axis=1)

imputed_complete = imputed_combined[imputed_combined['source'] == 'complete'].drop(columns=['source'])
imputed_incomplete = imputed_combined[imputed_combined['source'] == 'incomplete'].drop(columns=['source'])
imputed_incomplete = imputed_incomplete.reset_index(drop=True)

imputed_complete.to_excel("imputed_complete.xlsx", index=False)
imputed_incomplete.to_excel("imputed_incomplete.xlsx", index=False)

original_incomplete = incomplete_data.copy()
if 'source' in original_incomplete.columns:
    original_incomplete = original_incomplete.drop(columns=['source'])
original_incomplete = original_incomplete.reset_index(drop=True)
updated_incomplete = original_incomplete.copy()
for col in imputed_incomplete.columns:
    updated_incomplete[col] = imputed_incomplete[col]
updated_incomplete.to_excel("updated_incomplete_dataset.xlsx", index=False)

print("Updated incomplete dataset saved as 'updated_incomplete_dataset.xlsx'.")
