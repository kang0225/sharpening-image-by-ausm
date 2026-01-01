import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import params

def train_model(dataset_path: str):
    if not dataset_path or not os.path.exists(dataset_path):
        print("[INFO] 데이터셋이 존재하지 않습니다.")
        return
    
    print("[INFO] Started training...")
    df = pd.read_csv(dataset_path)
    X = df[['mean', 'std', 'high_freq_ratio']]
    y = df['target_k']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] 학습 데이터 : {len(X_train)}, 검증데이터: {len(X_val)}")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    print("[INFO] Training data...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    print(f"[INFO] Successfully completed. Test Set MSE: {mse:.6f}")
    
    model_path = os.path.join(params.output_dir, "xgb_model.joblib")
    joblib.dump(model, model_path)
    print(f"[INFO] Saved the model : {model_path}")

    return model, X_train, y_train, X_val, y_val

def evaluate(model, X_train, y_train, X_val, y_val):

    X_train = X_train.drop(columns=['target_k'], errors='ignore')
    X_val = X_val.drop(columns=['target_k'], errors='ignore')

    y_pred_train = model.predict(X_train) # 학습한데이터 예측
    y_pred_val = model.predict(X_val) # 새로운 데이터 예측

    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))    
    rmse_va = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmae_tr = np.sqrt(mean_absolute_error(y_train, y_pred_train))
    rmae_va = np.sqrt(mean_absolute_error(y_val, y_pred_val))
    
    print("========== Evaluate ==========")
    print("RMSE of Train Set : ", rmse_tr)
    print("RMSE of Test Set : ", rmse_va)
    print()
    print("RMAE of Train Set : ", rmae_tr)
    print("RMAE of Test Set : ", rmae_va)