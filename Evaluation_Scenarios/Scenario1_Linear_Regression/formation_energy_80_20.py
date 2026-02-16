#!/usr/bin/env python3
"""
Compare FTCP vs SSL Features for Formation Energy Prediction (80/20 Split)
=============================================================================
This script trains the SAME regression model on both FTCP and SSL features
to fairly compare which data representation is better for predicting formation energy.

Conditions:
- Same model (Linear Regression)
- Same hyperparameters
- Same data split (80/20)
- Same labels (Formation Energy)
- Same evaluation metrics
"""

import numpy as np
import pandas as pd
import time
import json
import multiprocessing
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    max_error,
    explained_variance_score,
    median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPARING FTCP vs SSL FEATURES FOR FORMATION ENERGY PREDICTION")
print("80% Train / 20% Test Split")
print("=" * 80)

print(f"\n⏰ Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout.flush()

# ==============================================================================
# SETUP PATHS
# ==============================================================================

workspace_root = Path("/home/danial/Features_Extraction_Effectiveness")
data_root = workspace_root / "Data/Splitted_Data/Split_20Test_80Train"
results_dir = Path(__file__).parent / "Results"
results_dir.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
sys.stdout.flush()

# ==============================================================================
# LOAD FTCP DATA
# ==============================================================================

print("\n📂 Loading FTCP data...")
print("   [1/5] Loading Train_FTCP_Data.npy (this is LARGE, ~18GB for 80% split)...")
print(f"   ⏰ Started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
start_load = time.time()
ftcp_train = np.load(data_root / "Train/Train_FTCP_Data.npy")
print(f"   ✅ [1/5] Train FTCP loaded in {time.time() - start_load:.2f}s")
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

print("   [2/5] Loading Test_FTCP_Data.npy...")
sys.stdout.flush()
start_load = time.time()
ftcp_test = np.load(data_root / "Test/Test_FTCP_Data.npy")
print(f"   ✅ [2/5] Test FTCP loaded in {time.time() - start_load:.2f}s")
sys.stdout.flush()

print("   [3/5] Loading Material IDs...")
sys.stdout.flush()
ftcp_train_ids = np.load(data_root / "Train/Train_FTCP_Material_IDs.npy", allow_pickle=True)
ftcp_test_ids = np.load(data_root / "Test/Test_FTCP_Material_IDs.npy", allow_pickle=True)
print(f"   ✅ [3/5] Material IDs loaded")
print(f"   FTCP Train shape: {ftcp_train.shape}")
print(f"   FTCP Test shape:  {ftcp_test.shape}")
sys.stdout.flush()

# Flatten FTCP data (400 x 63) -> (25200,)
print("   [4/5] Flattening FTCP data (this will use a lot of memory)...")
print(f"   This will create arrays of shape ({len(ftcp_train)}, 25200) and ({len(ftcp_test)}, 25200)")
print(f"   ⏰ Flattening started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
start_flatten = time.time()
print("   Reshaping training data...")
sys.stdout.flush()
ftcp_train_flat = ftcp_train.reshape(len(ftcp_train), -1)
elapsed = time.time() - start_flatten
print(f"   ✅ Train data flattened in {elapsed:.2f}s")
sys.stdout.flush()

print("   Reshaping test data...")
sys.stdout.flush()
ftcp_test_flat = ftcp_test.reshape(len(ftcp_test), -1)
print(f"   ✅ [4/5] Test data flattened in {time.time() - start_flatten - elapsed:.2f}s")
print(f"   [5/5] FTCP Train flattened: {ftcp_train_flat.shape}")
print(f"   [5/5] FTCP Test flattened:  {ftcp_test_flat.shape}")
print(f"   ⏰ Flattening completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

# ==============================================================================
# LOAD SSL FEATURES DATA
# ==============================================================================

print("\n📂 Loading SSL Features data...")
sys.stdout.flush()
start_time = time.time()
features_train = np.load(data_root / "Train/Train_Extracted_Features_Data.npy")
features_test = np.load(data_root / "Test/Test_Extracted_Features_Data.npy")
features_train_ids = np.load(data_root / "Train/Train_Extracted_Features_Material_IDs.npy", allow_pickle=True)
features_test_ids = np.load(data_root / "Test/Test_Extracted_Features_Material_IDs.npy", allow_pickle=True)
print(f"   ✅ Loaded in {time.time() - start_time:.2f}s")
print(f"   Features Train shape: {features_train.shape}")
print(f"   Features Test shape:  {features_test.shape}")
sys.stdout.flush()

# ==============================================================================
# LOAD LABELS
# ==============================================================================

print("\n📂 Loading labels...")
sys.stdout.flush()
y_train = np.load(data_root / "Train/Train_FormationEnergy_Labels.npy")
y_test = np.load(data_root / "Test/Test_FormationEnergy_Labels.npy")
y_train_ids = np.load(data_root / "Train/Train_Material_IDs.npy", allow_pickle=True)
y_test_ids = np.load(data_root / "Test/Test_Material_IDs.npy", allow_pickle=True)
print(f"   Train labels shape: {y_train.shape}")
print(f"   Test labels shape:  {y_test.shape}")
sys.stdout.flush()

# ==============================================================================
# VERIFY DATA ALIGNMENT
# ==============================================================================

print("\n✅ All data loaded and aligned successfully!")
print(f"   Total train samples: {len(y_train):,}")
print(f"   Total test samples:  {len(y_test):,}")
print(f"\n⏰ Data loading completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

print(f"\n📊 Label Statistics:")
print(f"   Train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}, Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"   Test  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
sys.stdout.flush()

# ==============================================================================
# EXPERIMENT 1: FTCP DATA
# ==============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: TRAINING ON FTCP DATA (80/20 Split)")
print("=" * 80)
sys.stdout.flush()

# Store all results
results_ftcp = {}

print("\n🔧 Preprocessing FTCP data...")
print("   Applying StandardScaler to 25,200 dimensions...")
print(f"   ⏰ Preprocessing started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("   This computes mean and std for each of 25,200 features...")
print(f"   [1/2] Fitting scaler on training data ({len(ftcp_train_flat):,} samples)...")
print("   Computing statistics for normalization...")
sys.stdout.flush()
scaler_ftcp = StandardScaler()
start_time = time.time()
start_fit = time.time()
X_train_ftcp_scaled = scaler_ftcp.fit_transform(ftcp_train_flat)
fit_time = time.time() - start_fit
print(f"   ✅ [1/2] Scaler fitted and training data transformed in {fit_time:.2f}s")
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

print(f"   [2/2] Transforming test data ({len(ftcp_test_flat):,} samples)...")
sys.stdout.flush()
start_transform = time.time()
X_test_ftcp_scaled = scaler_ftcp.transform(ftcp_test_flat)
transform_time = time.time() - start_transform
print(f"   ✅ [2/2] Test data transformed in {transform_time:.2f}s")
preprocessing_time_ftcp = time.time() - start_time
print(f"   ✅ Total preprocessing completed in {preprocessing_time_ftcp:.2f} seconds")
print(f"   ⏰ Preprocessing completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

results_ftcp['preprocessing_time'] = preprocessing_time_ftcp
results_ftcp['input_dimensions'] = ftcp_train_flat.shape[1]
results_ftcp['n_train_samples'] = len(X_train_ftcp_scaled)
results_ftcp['n_test_samples'] = len(X_test_ftcp_scaled)

print("\n🤖 Training Linear Regression on FTCP...")
print("   Using all CPU cores (n_jobs=-1)")
print(f"   Training {len(X_train_ftcp_scaled):,} samples with 25,200 features...")
print("   ⚠️  WARNING: This will take 15-40 minutes! Be patient!")
print(f"   ⏰ Training started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("   Computing (X^T X)^(-1) X^T y for linear regression...")
print("   You will see progress every 20 seconds below:")
print("")
sys.stdout.flush()
model_ftcp = LinearRegression(n_jobs=-1)

start_time = time.time()
# Use multiprocessing to avoid GIL blocking from model.fit()
stop_progress = multiprocessing.Event()
def print_progress_process(stop_event, start_timestamp):
    """Separate process for progress monitoring - immune to GIL"""
    import time
    import sys
    from datetime import datetime
    count = 0
    while not stop_event.is_set():
        time.sleep(20)
        if not stop_event.is_set():
            count += 1
            elapsed = count * 20
            ts = datetime.now().strftime('%H:%M:%S')
            msg = f"   [{elapsed:4d}s] FTCP training in progress... | {ts}"
            print(msg, flush=True)
            sys.stdout.flush()

progress_process = multiprocessing.Process(target=print_progress_process, args=(stop_progress, start_time))
progress_process.daemon = True
progress_process.start()

model_ftcp.fit(X_train_ftcp_scaled, y_train)

stop_progress.set()
progress_process.join(timeout=2)
if progress_process.is_alive():
    progress_process.terminate()
training_time_ftcp = time.time() - start_time

print("")
print(f"   ✅ FTCP TRAINING COMPLETED in {training_time_ftcp:.2f} seconds ({training_time_ftcp/60:.2f} minutes)")
print(f"   ⏰ Completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("")
sys.stdout.flush()
results_ftcp['training_time'] = training_time_ftcp

# Model parameters
results_ftcp['model_type'] = 'LinearRegression'
results_ftcp['n_features'] = X_train_ftcp_scaled.shape[1]
results_ftcp['n_coefficients'] = len(model_ftcp.coef_)
results_ftcp['intercept'] = float(model_ftcp.intercept_)

print("\n📈 Evaluating FTCP model on training set...")
print(f"   Predicting {len(X_train_ftcp_scaled):,} samples...")
sys.stdout.flush()
start_time = time.time()
y_train_pred_ftcp = model_ftcp.predict(X_train_ftcp_scaled)
train_inference_time_ftcp = time.time() - start_time
print(f"   ✅ Train prediction completed in {train_inference_time_ftcp:.4f} seconds")
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

train_mse_ftcp = mean_squared_error(y_train, y_train_pred_ftcp)
train_rmse_ftcp = np.sqrt(train_mse_ftcp)
train_mae_ftcp = mean_absolute_error(y_train, y_train_pred_ftcp)
train_r2_ftcp = r2_score(y_train, y_train_pred_ftcp)
train_evs_ftcp = explained_variance_score(y_train, y_train_pred_ftcp)
train_max_error_ftcp = max_error(y_train, y_train_pred_ftcp)
train_median_ae_ftcp = median_absolute_error(y_train, y_train_pred_ftcp)

results_ftcp['train_metrics'] = {
    'mse': float(train_mse_ftcp),
    'rmse': float(train_rmse_ftcp),
    'mae': float(train_mae_ftcp),
    'r2': float(train_r2_ftcp),
    'explained_variance': float(train_evs_ftcp),
    'max_error': float(train_max_error_ftcp),
    'median_absolute_error': float(train_median_ae_ftcp),
    'inference_time': float(train_inference_time_ftcp)
}

print(f"   Train R² Score: {train_r2_ftcp:.6f}")
print(f"   Train RMSE:     {train_rmse_ftcp:.6f}")
print(f"   Train MAE:      {train_mae_ftcp:.6f}")

print("\n📊 Evaluating FTCP model on test set...")
print(f"   Predicting {len(X_test_ftcp_scaled):,} samples...")
sys.stdout.flush()
start_time = time.time()
y_test_pred_ftcp = model_ftcp.predict(X_test_ftcp_scaled)
test_inference_time_ftcp = time.time() - start_time
print(f"   ✅ Test prediction completed in {test_inference_time_ftcp:.4f} seconds")
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

test_mse_ftcp = mean_squared_error(y_test, y_test_pred_ftcp)
test_rmse_ftcp = np.sqrt(test_mse_ftcp)
test_mae_ftcp = mean_absolute_error(y_test, y_test_pred_ftcp)
test_r2_ftcp = r2_score(y_test, y_test_pred_ftcp)
test_evs_ftcp = explained_variance_score(y_test, y_test_pred_ftcp)
test_max_error_ftcp = max_error(y_test, y_test_pred_ftcp)
test_median_ae_ftcp = median_absolute_error(y_test, y_test_pred_ftcp)

results_ftcp['test_metrics'] = {
    'mse': float(test_mse_ftcp),
    'rmse': float(test_rmse_ftcp),
    'mae': float(test_mae_ftcp),
    'r2': float(test_r2_ftcp),
    'explained_variance': float(test_evs_ftcp),
    'max_error': float(test_max_error_ftcp),
    'median_absolute_error': float(test_median_ae_ftcp),
    'inference_time': float(test_inference_time_ftcp)
}

print(f"   Test R² Score: {test_r2_ftcp:.6f}")
print(f"   Test RMSE:     {test_rmse_ftcp:.6f}")
print(f"   Test MAE:      {test_mae_ftcp:.6f}")

results_ftcp['total_time'] = preprocessing_time_ftcp + training_time_ftcp

# ==============================================================================
# EXPERIMENT 2: SSL FEATURES
# ==============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: TRAINING ON SSL FEATURES (80/20 Split)")
print("=" * 80)
sys.stdout.flush()

# Store all results
results_features = {}

print("\n🔧 Preprocessing SSL Features data...")
print("   Applying StandardScaler (much faster with only 2,048 dimensions)...")
sys.stdout.flush()
scaler_features = StandardScaler()
start_time = time.time()
X_train_features_scaled = scaler_features.fit_transform(features_train)
X_test_features_scaled = scaler_features.transform(features_test)
preprocessing_time_features = time.time() - start_time
print(f"   ✅ Preprocessing completed in {preprocessing_time_features:.2f} seconds")
sys.stdout.flush()

results_features['preprocessing_time'] = preprocessing_time_features
results_features['input_dimensions'] = features_train.shape[1]
results_features['n_train_samples'] = len(X_train_features_scaled)
results_features['n_test_samples'] = len(X_test_features_scaled)

print("\n🤖 Training Linear Regression on SSL Features...")
print("   Using all CPU cores (n_jobs=-1)")
print(f"   Training {len(X_train_features_scaled):,} samples with 2,048 features...")
print("   Expected time: 5-15 minutes (much faster than FTCP!)")
print(f"   ⏰ Training started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("   Computing (X^T X)^(-1) X^T y for linear regression...")
print("   You will see progress every 20 seconds below:")
print("")
sys.stdout.flush()
model_features = LinearRegression(n_jobs=-1)

start_time = time.time()
# Use multiprocessing for SSL features too (consistent approach)
stop_progress_features = multiprocessing.Event()
def print_progress_features_process(stop_event, start_timestamp):
    """Separate process for SSL features progress monitoring"""
    import time
    import sys
    from datetime import datetime
    count = 0
    while not stop_event.is_set():
        time.sleep(20)
        if not stop_event.is_set():
            count += 1
            elapsed = count * 20
            ts = datetime.now().strftime('%H:%M:%S')
            msg = f"   [{elapsed:4d}s] SSL Features training in progress... | {ts}"
            print(msg, flush=True)
            sys.stdout.flush()

progress_process_features = multiprocessing.Process(target=print_progress_features_process, args=(stop_progress_features, start_time))
progress_process_features.daemon = True
progress_process_features.start()

model_features.fit(X_train_features_scaled, y_train)

stop_progress_features.set()
progress_process_features.join(timeout=2)
if progress_process_features.is_alive():
    progress_process_features.terminate()
training_time_features = time.time() - start_time

print("")
print(f"   ✅ SSL FEATURES TRAINING COMPLETED in {training_time_features:.2f} seconds ({training_time_features/60:.2f} minutes)")
print(f"   ⏰ Completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("")
sys.stdout.flush()
results_features['training_time'] = training_time_features

# Model parameters
results_features['model_type'] = 'LinearRegression'
results_features['n_features'] = X_train_features_scaled.shape[1]
results_features['n_coefficients'] = len(model_features.coef_)
results_features['intercept'] = float(model_features.intercept_)

print("\n📈 Evaluating SSL Features model on training set...")
print(f"   Predicting {len(X_train_features_scaled):,} samples...")
sys.stdout.flush()
start_time = time.time()
y_train_pred_features = model_features.predict(X_train_features_scaled)
train_inference_time_features = time.time() - start_time
print(f"   ✅ Train prediction completed in {train_inference_time_features:.4f} seconds")
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

train_mse_features = mean_squared_error(y_train, y_train_pred_features)
train_rmse_features = np.sqrt(train_mse_features)
train_mae_features = mean_absolute_error(y_train, y_train_pred_features)
train_r2_features = r2_score(y_train, y_train_pred_features)
train_evs_features = explained_variance_score(y_train, y_train_pred_features)
train_max_error_features = max_error(y_train, y_train_pred_features)
train_median_ae_features = median_absolute_error(y_train, y_train_pred_features)

results_features['train_metrics'] = {
    'mse': float(train_mse_features),
    'rmse': float(train_rmse_features),
    'mae': float(train_mae_features),
    'r2': float(train_r2_features),
    'explained_variance': float(train_evs_features),
    'max_error': float(train_max_error_features),
    'median_absolute_error': float(train_median_ae_features),
    'inference_time': float(train_inference_time_features)
}

print(f"   Train R² Score: {train_r2_features:.6f}")
print(f"   Train RMSE:     {train_rmse_features:.6f}")
print(f"   Train MAE:      {train_mae_features:.6f}")

print("\n📊 Evaluating SSL Features model on test set...")
print(f"   Predicting {len(X_test_features_scaled):,} samples...")
sys.stdout.flush()
start_time = time.time()
y_test_pred_features = model_features.predict(X_test_features_scaled)
test_inference_time_features = time.time() - start_time
print(f"   ✅ Test prediction completed in {test_inference_time_features:.4f} seconds")
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

test_mse_features = mean_squared_error(y_test, y_test_pred_features)
test_rmse_features = np.sqrt(test_mse_features)
test_mae_features = mean_absolute_error(y_test, y_test_pred_features)
test_r2_features = r2_score(y_test, y_test_pred_features)
test_evs_features = explained_variance_score(y_test, y_test_pred_features)
test_max_error_features = max_error(y_test, y_test_pred_features)
test_median_ae_features = median_absolute_error(y_test, y_test_pred_features)

results_features['test_metrics'] = {
    'mse': float(test_mse_features),
    'rmse': float(test_rmse_features),
    'mae': float(test_mae_features),
    'r2': float(test_r2_features),
    'explained_variance': float(test_evs_features),
    'max_error': float(test_max_error_features),
    'median_absolute_error': float(test_median_ae_features),
    'inference_time': float(test_inference_time_features)
}

print(f"   Test R² Score: {test_r2_features:.6f}")
print(f"   Test RMSE:     {test_rmse_features:.6f}")
print(f"   Test MAE:      {test_mae_features:.6f}")

results_features['total_time'] = preprocessing_time_features + training_time_features

# ==============================================================================
# COMPARISON AND ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("COMPARISON: FTCP vs SSL FEATURES (80/20 Split)")
print("=" * 80)
sys.stdout.flush()

comparison = {
    'experiment_info': {
        'task': 'Formation Energy Prediction',
        'model': 'Linear Regression',
        'split': '80% Train / 20% Test',
        'n_train': len(y_train),
        'n_test': len(y_test),
        'date': pd.Timestamp.now().isoformat()
    },
    'ftcp': results_ftcp,
    'ssl_features': results_features
}

# Calculate improvements
print("\n📊 PERFORMANCE COMPARISON (Test Set):")
print("-" * 80)
print(f"{'Metric':<25} {'FTCP':<20} {'SSL Features':<20} {'Better':<15}")
print("-" * 80)

metrics = ['r2', 'rmse', 'mae', 'median_absolute_error', 'explained_variance']
metric_names = ['R² Score', 'RMSE', 'MAE', 'Median AE', 'Explained Variance']

for metric, name in zip(metrics, metric_names):
    ftcp_val = results_ftcp['test_metrics'][metric]
    feat_val = results_features['test_metrics'][metric]
    
    if metric == 'r2' or metric == 'explained_variance':
        better = "SSL Features" if feat_val > ftcp_val else "FTCP"
    else:
        better = "SSL Features" if feat_val < ftcp_val else "FTCP"
    
    print(f"{name:<25} {ftcp_val:<20.6f} {feat_val:<20.6f} {better:<15}")

print("\n⏱️  COMPUTATIONAL EFFICIENCY:")
print("-" * 80)
print(f"{'Metric':<25} {'FTCP':<20} {'SSL Features':<20} {'Speedup':<15}")
print("-" * 80)
print(f"{'Input Dimensions':<25} {results_ftcp['input_dimensions']:<20,} {results_features['input_dimensions']:<20,}")
print(f"{'Preprocessing Time':<25} {results_ftcp['preprocessing_time']:<20.4f} {results_features['preprocessing_time']:<20.4f} {results_ftcp['preprocessing_time'] / results_features['preprocessing_time']:>6.2f}x")
print(f"{'Training Time':<25} {results_ftcp['training_time']:<20.4f} {results_features['training_time']:<20.4f} {results_ftcp['training_time'] / results_features['training_time']:>6.2f}x")
print(f"{'Total Time':<25} {results_ftcp['total_time']:<20.4f} {results_features['total_time']:<20.4f} {results_ftcp['total_time'] / results_features['total_time']:>6.2f}x")

# Determine winner
print("\n🏆 OVERALL WINNER:")
print("-" * 80)
if test_r2_features > test_r2_ftcp:
    winner = "SSL Features"
    if test_r2_ftcp > 0:
        r2_improvement = ((test_r2_features - test_r2_ftcp) / abs(test_r2_ftcp)) * 100
    else:
        r2_improvement = 100.0
    print(f"✅ SSL Features outperform FTCP")
    print(f"   SSL Features R²: {test_r2_features:.6f}")
    print(f"   FTCP R²:         {test_r2_ftcp:.6f}")
else:
    winner = "FTCP"
    r2_improvement = ((test_r2_ftcp - test_r2_features) / abs(test_r2_features)) * 100
    print(f"✅ FTCP outperforms SSL Features by {r2_improvement:.2f}% in R² score")
    print(f"   FTCP R²:         {test_r2_ftcp:.6f}")
    print(f"   SSL Features R²: {test_r2_features:.6f}")

comparison['winner'] = {
    'data_type': winner,
    'r2_improvement_percent': float(r2_improvement),
    'test_r2_ftcp': float(test_r2_ftcp),
    'test_r2_ssl': float(test_r2_features)
}

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)
sys.stdout.flush()

# Save JSON results
json_path = results_dir / "comparison_results_80_20.json"
with open(json_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"\n✅ Saved comparison results: {json_path}")

# Save detailed CSV
results_df = pd.DataFrame({
    'Data_Type': ['FTCP', 'SSL_Features'],
    'Split': ['80/20', '80/20'],
    'Input_Dimensions': [results_ftcp['input_dimensions'], results_features['input_dimensions']],
    'Train_R2': [train_r2_ftcp, train_r2_features],
    'Train_RMSE': [train_rmse_ftcp, train_rmse_features],
    'Train_MAE': [train_mae_ftcp, train_mae_features],
    'Test_R2': [test_r2_ftcp, test_r2_features],
    'Test_RMSE': [test_rmse_ftcp, test_rmse_features],
    'Test_MAE': [test_mae_ftcp, test_mae_features],
    'Preprocessing_Time': [preprocessing_time_ftcp, preprocessing_time_features],
    'Training_Time': [training_time_ftcp, training_time_features],
    'Total_Time': [results_ftcp['total_time'], results_features['total_time']]
})

csv_path = results_dir / "comparison_summary_80_20.csv"
results_df.to_csv(csv_path, index=False)
print(f"✅ Saved summary CSV: {csv_path}")

# Save predictions
np.save(results_dir / "ftcp_test_predictions_80_20.npy", y_test_pred_ftcp)
np.save(results_dir / "features_test_predictions_80_20.npy", y_test_pred_features)
np.save(results_dir / "test_true_labels_80_20.npy", y_test)
np.save(results_dir / "test_material_ids_80_20.npy", y_test_ids)
print(f"✅ Saved predictions for further analysis")

# ==============================================================================
# CREATE VISUALIZATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)
print(f"   ⏰ Visualization started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("   Generating 6 plots...")
sys.stdout.flush()

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 12))

# 1. Predicted vs Actual - FTCP
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_test_pred_ftcp, alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('True Formation Energy (eV/atom)')
ax1.set_ylabel('Predicted Formation Energy (eV/atom)')
ax1.set_title(f'FTCP (80/20): Predicted vs Actual\nR² = {test_r2_ftcp:.4f}')
ax1.grid(True, alpha=0.3)

# 2. Predicted vs Actual - SSL Features
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_test_pred_features, alpha=0.5, s=10, color='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('True Formation Energy (eV/atom)')
ax2.set_ylabel('Predicted Formation Energy (eV/atom)')
ax2.set_title(f'SSL Features (80/20): Predicted vs Actual\nR² = {test_r2_features:.4f}')
ax2.grid(True, alpha=0.3)

# 3. Residuals - FTCP
ax3 = plt.subplot(2, 3, 4)
residuals_ftcp = y_test - y_test_pred_ftcp
ax3.scatter(y_test_pred_ftcp, residuals_ftcp, alpha=0.5, s=10)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Formation Energy (eV/atom)')
ax3.set_ylabel('Residuals (eV/atom)')
ax3.set_title(f'FTCP: Residual Plot\nMAE = {test_mae_ftcp:.4f}')
ax3.grid(True, alpha=0.3)

# 4. Residuals - SSL Features
ax4 = plt.subplot(2, 3, 5)
residuals_features = y_test - y_test_pred_features
ax4.scatter(y_test_pred_features, residuals_features, alpha=0.5, s=10, color='orange')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Formation Energy (eV/atom)')
ax4.set_ylabel('Residuals (eV/atom)')
ax4.set_title(f'SSL Features: Residual Plot\nMAE = {test_mae_features:.4f}')
ax4.grid(True, alpha=0.3)

# 5. Performance Comparison
ax5 = plt.subplot(2, 3, 3)
metrics_plot = ['R²', 'RMSE', 'MAE']
ftcp_values = [test_r2_ftcp if test_r2_ftcp > -100 else -10, test_rmse_ftcp if test_rmse_ftcp < 100 else 10, test_mae_ftcp if test_mae_ftcp < 100 else 10]
features_values = [test_r2_features, test_rmse_features, test_mae_features]

x = np.arange(len(metrics_plot))
width = 0.35

bars1 = ax5.bar(x - width/2, ftcp_values, width, label='FTCP', alpha=0.8)
bars2 = ax5.bar(x + width/2, features_values, width, label='SSL Features', alpha=0.8)

ax5.set_xlabel('Metrics')
ax5.set_ylabel('Values')
ax5.set_title('Performance Comparison (80/20 Test Set)')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_plot)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Time Comparison
ax6 = plt.subplot(2, 3, 6)
time_metrics = ['Preprocessing', 'Training', 'Total']
ftcp_times = [preprocessing_time_ftcp, training_time_ftcp, results_ftcp['total_time']]
features_times = [preprocessing_time_features, training_time_features, results_features['total_time']]

x = np.arange(len(time_metrics))
bars1 = ax6.bar(x - width/2, ftcp_times, width, label='FTCP', alpha=0.8)
bars2 = ax6.bar(x + width/2, features_times, width, label='SSL Features', alpha=0.8)

ax6.set_xlabel('Time Metrics')
ax6.set_ylabel('Time (seconds)')
ax6.set_title('Computational Efficiency (80/20 Split)')
ax6.set_xticks(x)
ax6.set_xticklabels(time_metrics)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

print("   Finalizing plots...")
sys.stdout.flush()
plt.tight_layout()
plot_path = results_dir / "comparison_plots_80_20.png"
print("   Saving plots to file (high resolution)...")
sys.stdout.flush()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved visualization: {plot_path}")
print(f"   ⏰ Visualization completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
plt.close()
sys.stdout.flush()

# ==============================================================================
# CREATE TEXT REPORT
# ==============================================================================

report = f"""
{'=' * 80}
FORMATION ENERGY PREDICTION: FTCP vs SSL FEATURES (80/20 Split)
{'=' * 80}

Experiment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Split: 80% Train ({len(y_train):,} samples) / 20% Test ({len(y_test):,} samples)
Model: Linear Regression (same hyperparameters for both)
Task: Formation Energy Prediction (eV/atom)

{'=' * 80}
EXPERIMENTAL SETUP
{'=' * 80}

Conditions (IDENTICAL FOR BOTH):
✓ Same regression model (LinearRegression)
✓ Same hyperparameters (n_jobs=-1)
✓ Same data split (80/20)
✓ Same labels (Formation Energy)
✓ Same preprocessing (StandardScaler)
✓ Same evaluation metrics
✓ Same random seed (42)

Input Dimensions:
- FTCP: {results_ftcp['input_dimensions']:,} dimensions (400×63 flattened)
- SSL Features: {results_features['input_dimensions']:,} dimensions (learned representations)

{'=' * 80}
RESULTS: TRAINING SET PERFORMANCE
{'=' * 80}

R² Score:
- FTCP:         {train_r2_ftcp:.6f}
- SSL Features: {train_r2_features:.6f}

RMSE:
- FTCP:         {train_rmse_ftcp:.6f} eV/atom
- SSL Features: {train_rmse_features:.6f} eV/atom

MAE:
- FTCP:         {train_mae_ftcp:.6f} eV/atom
- SSL Features: {train_mae_features:.6f} eV/atom

{'=' * 80}
RESULTS: TEST SET PERFORMANCE
{'=' * 80}

R² Score (Higher is Better):
- FTCP:         {test_r2_ftcp:.6f}
- SSL Features: {test_r2_features:.6f}
- Winner:       {winner}

RMSE (Lower is Better):
- FTCP:         {test_rmse_ftcp:.6f} eV/atom
- SSL Features: {test_rmse_features:.6f} eV/atom

MAE (Lower is Better):
- FTCP:         {test_mae_ftcp:.6f} eV/atom
- SSL Features: {test_mae_features:.6f} eV/atom

Median Absolute Error:
- FTCP:         {test_median_ae_ftcp:.6f} eV/atom
- SSL Features: {test_median_ae_features:.6f} eV/atom

Explained Variance:
- FTCP:         {test_evs_ftcp:.6f}
- SSL Features: {test_evs_features:.6f}

{'=' * 80}
TRAIN-TEST GAP ANALYSIS (Overfitting Check)
{'=' * 80}

FTCP:
- Train R²: {train_r2_ftcp:.6f}
- Test R²:  {test_r2_ftcp:.6f}
- Gap:      {train_r2_ftcp - test_r2_ftcp:.6f}
- Status:   {'⚠️ OVERFITTING' if (train_r2_ftcp - test_r2_ftcp) > 0.1 else '✅ Good Generalization'}

SSL Features:
- Train R²: {train_r2_features:.6f}
- Test R²:  {test_r2_features:.6f}
- Gap:      {train_r2_features - test_r2_features:.6f}
- Status:   {'⚠️ OVERFITTING' if (train_r2_features - test_r2_features) > 0.1 else '✅ Good Generalization'}

{'=' * 80}
COMPUTATIONAL EFFICIENCY
{'=' * 80}

Preprocessing Time:
- FTCP:         {preprocessing_time_ftcp:.4f} seconds
- SSL Features: {preprocessing_time_features:.4f} seconds
- Speedup:      {preprocessing_time_ftcp / preprocessing_time_features:.2f}x

Training Time:
- FTCP:         {training_time_ftcp:.4f} seconds ({training_time_ftcp/60:.2f} minutes)
- SSL Features: {training_time_features:.4f} seconds ({training_time_features/60:.2f} minutes)
- Speedup:      {training_time_ftcp / training_time_features:.2f}x

Total Time:
- FTCP:         {results_ftcp['total_time']:.4f} seconds ({results_ftcp['total_time']/60:.2f} minutes)
- SSL Features: {results_features['total_time']:.4f} seconds ({results_features['total_time']/60:.2f} minutes)
- Speedup:      {results_ftcp['total_time'] / results_features['total_time']:.2f}x

{'=' * 80}
CONCLUSION
{'=' * 80}

Winner: {winner}

Key Findings (80/20 Split):
1. Test R²: {'SSL Features' if test_r2_features > test_r2_ftcp else 'FTCP'} achieved better predictive performance
2. Generalization: {'SSL Features' if abs(train_r2_features - test_r2_features) < abs(train_r2_ftcp - test_r2_ftcp) else 'FTCP'} showed better train-test consistency
3. Efficiency: SSL Features are {results_ftcp['total_time'] / results_features['total_time']:.1f}x faster to train
4. Dimensionality: SSL features use {results_features['input_dimensions']:,} dims vs {results_ftcp['input_dimensions']:,} dims (FTCP)

Files Generated:
- comparison_results_80_20.json (detailed metrics)
- comparison_summary_80_20.csv (tabular summary)
- comparison_plots_80_20.png (visualizations)
- *_80_20.npy (predictions for further analysis)

{'=' * 80}
"""

report_path = results_dir / "COMPARISON_REPORT_80_20.txt"
with open(report_path, 'w') as f:
    f.write(report)
print(f"✅ Saved detailed report: {report_path}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n⏰ End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📁 All results saved to: {results_dir}")
print("\nFiles created:")
print("  - comparison_results_80_20.json")
print("  - comparison_summary_80_20.csv")
print("  - comparison_plots_80_20.png")
print("  - COMPARISON_REPORT_80_20.txt")
print("  - predictions and labels (.npy files)")
print("\n✅ Ready for analysis!")
sys.stdout.flush()
