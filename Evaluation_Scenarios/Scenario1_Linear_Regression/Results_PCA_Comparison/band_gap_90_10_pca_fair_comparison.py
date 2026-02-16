#!/usr/bin/env python3
"""
Fair Comparison: FTCP+PCA vs SSL Features for Band Gap (90/10 Split)
======================================================================
This script provides a FAIRER comparison by reducing FTCP dimensionality to
match SSL (2,048D) using PCA before training Linear Regression.

Key Question: Is SSL's advantage due to dimensionality reduction alone,
or do SSL features capture better representations than PCA?
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
from sklearn.decomposition import PCA
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
print("FAIR COMPARISON: FTCP+PCA vs SSL FEATURES")
print("Band Gap Prediction (90/10 Split)")
print("Both reduced to 2,048 dimensions for fair comparison")
print("=" * 80)

print(f"\n⏰ Start time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout.flush()

# ==============================================================================
# SETUP PATHS
# ==============================================================================

workspace_root = Path("/home/danial/Features_Extraction_Effectiveness")
data_root = workspace_root / "Data/Splitted_Data/Split_10Test_90Train"
results_dir = Path(__file__).parent / "Results_PCA_Comparison"
results_dir.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
sys.stdout.flush()

# ==============================================================================
# LOAD FTCP DATA
# ==============================================================================

print("\n📂 Loading FTCP data...")
print("   [1/5] Loading Train_FTCP_Data.npy (~22GB for 90% split)...")
print(f"   ⏰ Started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
start_load = time.time()
ftcp_train = np.load(data_root / "Train/Train_FTCP_Data.npy")
print(f"   ✅ [1/5] Train FTCP loaded in {time.time() - start_load:.2f}s")
sys.stdout.flush()

print("   [2/5] Loading Test_FTCP_Data.npy...")
sys.stdout.flush()
start_load = time.time()
ftcp_test = np.load(data_root / "Test/Test_FTCP_Data.npy")
print(f"   ✅ [2/5] Test FTCP loaded in {time.time() - start_load:.2f}s")
sys.stdout.flush()

print("   [3/5] Flattening FTCP data...")
sys.stdout.flush()
ftcp_train_flat = ftcp_train.reshape(len(ftcp_train), -1)
ftcp_test_flat = ftcp_test.reshape(len(ftcp_test), -1)
print(f"   ✅ [3/5] FTCP flattened: Train {ftcp_train_flat.shape}, Test {ftcp_test_flat.shape}")
sys.stdout.flush()

# ==============================================================================
# LOAD SSL FEATURES DATA
# ==============================================================================

print("\n📂 Loading SSL Features data...")
sys.stdout.flush()
start_time = time.time()
features_train = np.load(data_root / "Train/Train_Extracted_Features_Data.npy")
features_test = np.load(data_root / "Test/Test_Extracted_Features_Data.npy")
print(f"   ✅ Loaded in {time.time() - start_time:.2f}s")
print(f"   SSL Features: Train {features_train.shape}, Test {features_test.shape}")
sys.stdout.flush()

# ==============================================================================
# LOAD LABELS (BAND GAP)
# ==============================================================================

print("\n📂 Loading Band Gap labels...")
sys.stdout.flush()
y_train = np.load(data_root / "Train/Train_BandGap_Labels.npy")
y_test = np.load(data_root / "Test/Test_BandGap_Labels.npy")
print(f"   Train labels shape: {y_train.shape}")
print(f"   Test labels shape:  {y_test.shape}")
print(f"   Train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}, Mean: {y_train.mean():.4f} eV")
print(f"   Test  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f} eV")
sys.stdout.flush()

print("\n✅ All data loaded successfully!")
print(f"   Total train samples: {len(y_train):,}")
print(f"   Total test samples:  {len(y_test):,}")
sys.stdout.flush()

# ==============================================================================
# EXPERIMENT 1: FTCP+PCA (DIMENSIONALITY REDUCTION TO 2,048)
# ==============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: FTCP + PCA (Reduced to 2,048 dimensions)")
print("=" * 80)
sys.stdout.flush()

results_ftcp_pca = {}

print("\n🔧 Step 1: Applying StandardScaler to FTCP (25,200D)...")
sys.stdout.flush()
scaler_ftcp = StandardScaler()
start_time = time.time()
ftcp_train_scaled = scaler_ftcp.fit_transform(ftcp_train_flat)
ftcp_test_scaled = scaler_ftcp.transform(ftcp_test_flat)
scaling_time = time.time() - start_time
print(f"   ✅ Scaling completed in {scaling_time:.2f} seconds")
sys.stdout.flush()

print("\n🔬 Step 2: Applying PCA to reduce from 25,200D to 2,048D...")
print(f"   ⏰ PCA started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("   This will take 2-5 minutes...")
print("")
sys.stdout.flush()

pca = PCA(n_components=2048, random_state=42)
start_pca = time.time()

# PCA progress monitoring
stop_progress = multiprocessing.Event()
def print_pca_progress(stop_event, start_timestamp):
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
            msg = f"   [{elapsed:4d}s] PCA fitting in progress... | {ts}"
            print(msg, flush=True)
            sys.stdout.flush()

progress_process = multiprocessing.Process(target=print_pca_progress, args=(stop_progress, start_pca))
progress_process.daemon = True
progress_process.start()

X_train_ftcp_pca = pca.fit_transform(ftcp_train_scaled)

stop_progress.set()
progress_process.join(timeout=2)
if progress_process.is_alive():
    progress_process.terminate()

print("")
print(f"   ✅ PCA fitting completed in {time.time() - start_pca:.2f} seconds")
sys.stdout.flush()

print("   Transforming test data with fitted PCA...")
sys.stdout.flush()
start_transform = time.time()
X_test_ftcp_pca = pca.transform(ftcp_test_scaled)
transform_time = time.time() - start_transform
print(f"   ✅ Test transformation completed in {transform_time:.2f} seconds")
sys.stdout.flush()

pca_time = time.time() - start_pca + transform_time
preprocessing_time_ftcp_pca = scaling_time + pca_time

print(f"\n📊 PCA Statistics:")
print(f"   Explained variance:  {pca.explained_variance_ratio_.sum():.6f} ({pca.explained_variance_ratio_.sum()*100:.4f}%)")
print(f"   Total preprocessing time: {preprocessing_time_ftcp_pca:.2f} seconds")
sys.stdout.flush()

results_ftcp_pca['preprocessing_time'] = preprocessing_time_ftcp_pca
results_ftcp_pca['pca_time'] = pca_time
results_ftcp_pca['explained_variance'] = float(pca.explained_variance_ratio_.sum())

print("\n🤖 Training Linear Regression on FTCP+PCA (2,048D)...")
print(f"   ⏰ Training started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

model_ftcp_pca = LinearRegression(n_jobs=-1)
start_time = time.time()
model_ftcp_pca.fit(X_train_ftcp_pca, y_train)
training_time_ftcp_pca = time.time() - start_time

print(f"   ✅ TRAINING COMPLETED in {training_time_ftcp_pca:.2f} seconds")
sys.stdout.flush()
results_ftcp_pca['training_time'] = training_time_ftcp_pca

# Evaluate
y_train_pred_ftcp_pca = model_ftcp_pca.predict(X_train_ftcp_pca)
y_test_pred_ftcp_pca = model_ftcp_pca.predict(X_test_ftcp_pca)

train_r2_ftcp_pca = r2_score(y_train, y_train_pred_ftcp_pca)
train_rmse_ftcp_pca = np.sqrt(mean_squared_error(y_train, y_train_pred_ftcp_pca))
train_mae_ftcp_pca = mean_absolute_error(y_train, y_train_pred_ftcp_pca)

test_r2_ftcp_pca = r2_score(y_test, y_test_pred_ftcp_pca)
test_rmse_ftcp_pca = np.sqrt(mean_squared_error(y_test, y_test_pred_ftcp_pca))
test_mae_ftcp_pca = mean_absolute_error(y_test, y_test_pred_ftcp_pca)

results_ftcp_pca['train_metrics'] = {
    'r2': float(train_r2_ftcp_pca),
    'rmse': float(train_rmse_ftcp_pca),
    'mae': float(train_mae_ftcp_pca)
}

results_ftcp_pca['test_metrics'] = {
    'r2': float(test_r2_ftcp_pca),
    'rmse': float(test_rmse_ftcp_pca),
    'mae': float(test_mae_ftcp_pca)
}

print(f"\n📊 FTCP+PCA Results:")
print(f"   Train - R²: {train_r2_ftcp_pca:.6f}, RMSE: {train_rmse_ftcp_pca:.6f} eV, MAE: {train_mae_ftcp_pca:.6f} eV")
print(f"   Test  - R²: {test_r2_ftcp_pca:.6f}, RMSE: {test_rmse_ftcp_pca:.6f} eV, MAE: {test_mae_ftcp_pca:.6f} eV")
sys.stdout.flush()

results_ftcp_pca['total_time'] = preprocessing_time_ftcp_pca + training_time_ftcp_pca

# ==============================================================================
# EXPERIMENT 2: SSL FEATURES (2,048D)
# ==============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: SSL FEATURES (2,048 dimensions)")
print("=" * 80)
sys.stdout.flush()

results_ssl = {}

print("\n🔧 Preprocessing SSL Features...")
sys.stdout.flush()
scaler_ssl = StandardScaler()
start_time = time.time()
X_train_ssl_scaled = scaler_ssl.fit_transform(features_train)
X_test_ssl_scaled = scaler_ssl.transform(features_test)
preprocessing_time_ssl = time.time() - start_time
print(f"   ✅ Preprocessing completed in {preprocessing_time_ssl:.2f} seconds")
sys.stdout.flush()

results_ssl['preprocessing_time'] = preprocessing_time_ssl

print("\n🤖 Training Linear Regression on SSL Features...")
print(f"   ⏰ Training started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

model_ssl = LinearRegression(n_jobs=-1)
start_time = time.time()
model_ssl.fit(X_train_ssl_scaled, y_train)
training_time_ssl = time.time() - start_time

print(f"   ✅ TRAINING COMPLETED in {training_time_ssl:.2f} seconds")
sys.stdout.flush()
results_ssl['training_time'] = training_time_ssl

# Evaluate
y_train_pred_ssl = model_ssl.predict(X_train_ssl_scaled)
y_test_pred_ssl = model_ssl.predict(X_test_ssl_scaled)

train_r2_ssl = r2_score(y_train, y_train_pred_ssl)
train_rmse_ssl = np.sqrt(mean_squared_error(y_train, y_train_pred_ssl))
train_mae_ssl = mean_absolute_error(y_train, y_train_pred_ssl)

test_r2_ssl = r2_score(y_test, y_test_pred_ssl)
test_rmse_ssl = np.sqrt(mean_squared_error(y_test, y_test_pred_ssl))
test_mae_ssl = mean_absolute_error(y_test, y_test_pred_ssl)

results_ssl['train_metrics'] = {
    'r2': float(train_r2_ssl),
    'rmse': float(train_rmse_ssl),
    'mae': float(train_mae_ssl)
}

results_ssl['test_metrics'] = {
    'r2': float(test_r2_ssl),
    'rmse': float(test_rmse_ssl),
    'mae': float(test_mae_ssl)
}

print(f"\n📊 SSL Results:")
print(f"   Train - R²: {train_r2_ssl:.6f}, RMSE: {train_rmse_ssl:.6f} eV, MAE: {train_mae_ssl:.6f} eV")
print(f"   Test  - R²: {test_r2_ssl:.6f}, RMSE: {test_rmse_ssl:.6f} eV, MAE: {test_mae_ssl:.6f} eV")
sys.stdout.flush()

results_ssl['total_time'] = preprocessing_time_ssl + training_time_ssl

# ==============================================================================
# COMPARISON
# ==============================================================================

print("\n" + "=" * 80)
print("COMPARISON: FTCP+PCA vs SSL (Both 2,048D)")
print("=" * 80)
sys.stdout.flush()

comparison = {
    'experiment_info': {
        'task': 'Band Gap Prediction',
        'model': 'Linear Regression',
        'split': '90% Train / 10% Test',
        'n_train': len(y_train),
        'n_test': len(y_test),
        'dimensions': 2048,
        'date': pd.Timestamp.now().isoformat()
    },
    'ftcp_pca': results_ftcp_pca,
    'ssl': results_ssl
}

print("\n📊 TEST SET COMPARISON:")
print("-" * 80)
print(f"{'Metric':<20} {'FTCP+PCA':<20} {'SSL':<20} {'Winner':<15}")
print("-" * 80)
print(f"{'R²':<20} {test_r2_ftcp_pca:<20.6f} {test_r2_ssl:<20.6f} {'SSL' if test_r2_ssl > test_r2_ftcp_pca else 'FTCP+PCA':<15}")
print(f"{'RMSE (eV)':<20} {test_rmse_ftcp_pca:<20.6f} {test_rmse_ssl:<20.6f} {'SSL' if test_rmse_ssl < test_rmse_ftcp_pca else 'FTCP+PCA':<15}")
print(f"{'MAE (eV)':<20} {test_mae_ftcp_pca:<20.6f} {test_mae_ssl:<20.6f} {'SSL' if test_mae_ssl < test_mae_ftcp_pca else 'FTCP+PCA':<15}")

if test_r2_ssl > test_r2_ftcp_pca:
    winner = "SSL"
    r2_diff = test_r2_ssl - test_r2_ftcp_pca
else:
    winner = "FTCP+PCA"
    r2_diff = test_r2_ftcp_pca - test_r2_ssl

comparison['winner'] = {
    'model': winner,
    'r2_difference': float(abs(r2_diff))
}

print(f"\n🏆 WINNER: {winner}")
print(f"   R² difference: {abs(r2_diff):.6f}")
print(f"   RMSE difference: {abs(test_rmse_ssl - test_rmse_ftcp_pca):.6f} eV")
sys.stdout.flush()

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)
sys.stdout.flush()

json_path = results_dir / "pca_fair_comparison_band_gap_90_10.json"
with open(json_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"✅ Saved results: {json_path}")

np.save(results_dir / "ftcp_pca_test_predictions_band_gap_90_10.npy", y_test_pred_ftcp_pca)
np.save(results_dir / "ssl_test_predictions_band_gap_90_10.npy", y_test_pred_ssl)
np.save(results_dir / "test_true_labels_band_gap_90_10.npy", y_test)
print(f"✅ Saved predictions")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED!")
print("=" * 80)
print(f"⏰ End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Results saved to: {results_dir}")
sys.stdout.flush()
