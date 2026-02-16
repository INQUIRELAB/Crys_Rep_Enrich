#!/usr/bin/env python3
"""
Fair Comparison: FTCP+PCA vs SSL Features for Formation Energy (90/10 Split)
===============================================================================
This script provides a FAIRER comparison by reducing FTCP dimensionality to
match SSL (2,048D) using PCA before training Linear Regression.

Previous comparison:
- FTCP: 25,200 dimensions (catastrophic failure)
- SSL: 2,048 dimensions (success)
- Unfair due to dimensionality mismatch

THIS comparison:
- FTCP+PCA: 2,048 dimensions (reduced via PCA)
- SSL: 2,048 dimensions (learned representations)
- Fair: Same dimensionality, tests quality of representations

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
print("Formation Energy Prediction (90/10 Split)")
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
print("   [4/5] Flattening FTCP data...")
print(f"   This will create arrays of shape ({len(ftcp_train)}, 25200) and ({len(ftcp_test)}, 25200)")
print(f"   ⏰ Flattening started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
start_flatten = time.time()
ftcp_train_flat = ftcp_train.reshape(len(ftcp_train), -1)
ftcp_test_flat = ftcp_test.reshape(len(ftcp_test), -1)
print(f"   ✅ [4/5] Flattening completed in {time.time() - start_flatten:.2f}s")
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
print(f"   SSL Features Train shape: {features_train.shape}")
print(f"   SSL Features Test shape:  {features_test.shape}")
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

print("\n✅ All data loaded successfully!")
print(f"   Total train samples: {len(y_train):,}")
print(f"   Total test samples:  {len(y_test):,}")
print(f"\n⏰ Data loading completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

print(f"\n📊 Label Statistics:")
print(f"   Train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}, Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"   Test  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
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
print(f"   ⏰ Scaling started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
scaler_ftcp = StandardScaler()
start_time = time.time()
ftcp_train_scaled = scaler_ftcp.fit_transform(ftcp_train_flat)
ftcp_test_scaled = scaler_ftcp.transform(ftcp_test_flat)
scaling_time = time.time() - start_time
print(f"   ✅ Scaling completed in {scaling_time:.2f} seconds")
sys.stdout.flush()

print("\n🔬 Step 2: Applying PCA to reduce from 25,200D to 2,048D...")
print(f"   Target dimensions: 2,048 (same as SSL)")
print(f"   ⏰ PCA started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
print("   This will take 2-5 minutes...")
print("")
sys.stdout.flush()

pca = PCA(n_components=2048, random_state=42)
start_pca = time.time()

# PCA progress monitoring
stop_progress = multiprocessing.Event()
def print_pca_progress(stop_event, start_timestamp):
    """Monitor PCA progress"""
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
print(f"   ⏰ Current time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

print("\n   Transforming test data with fitted PCA...")
sys.stdout.flush()
start_transform = time.time()
X_test_ftcp_pca = pca.transform(ftcp_test_scaled)
transform_time = time.time() - start_transform
print(f"   ✅ Test transformation completed in {transform_time:.2f} seconds")
sys.stdout.flush()

pca_time = time.time() - start_pca + transform_time
preprocessing_time_ftcp_pca = scaling_time + pca_time

print(f"\n📊 PCA Statistics:")
print(f"   Original dimensions: 25,200")
print(f"   Reduced dimensions:  2,048")
print(f"   Explained variance:  {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
print(f"   Dimensionality reduction: {25200/2048:.1f}x")
print(f"   Total preprocessing time: {preprocessing_time_ftcp_pca:.2f} seconds")
sys.stdout.flush()

results_ftcp_pca['preprocessing_time'] = preprocessing_time_ftcp_pca
results_ftcp_pca['scaling_time'] = scaling_time
results_ftcp_pca['pca_time'] = pca_time
results_ftcp_pca['original_dimensions'] = 25200
results_ftcp_pca['reduced_dimensions'] = 2048
results_ftcp_pca['explained_variance'] = float(pca.explained_variance_ratio_.sum())
results_ftcp_pca['n_train_samples'] = len(X_train_ftcp_pca)
results_ftcp_pca['n_test_samples'] = len(X_test_ftcp_pca)

print("\n🤖 Training Linear Regression on FTCP+PCA (2,048D)...")
print(f"   Training {len(X_train_ftcp_pca):,} samples with 2,048 features...")
print(f"   Expected time: 30-60 seconds (much faster than 25,200D!)")
print(f"   ⏰ Training started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

model_ftcp_pca = LinearRegression(n_jobs=-1)
start_time = time.time()
model_ftcp_pca.fit(X_train_ftcp_pca, y_train)
training_time_ftcp_pca = time.time() - start_time

print(f"   ✅ TRAINING COMPLETED in {training_time_ftcp_pca:.2f} seconds")
print(f"   ⏰ Completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
results_ftcp_pca['training_time'] = training_time_ftcp_pca

# Evaluate on train set
print("\n📈 Evaluating on training set...")
sys.stdout.flush()
y_train_pred_ftcp_pca = model_ftcp_pca.predict(X_train_ftcp_pca)
train_mse_ftcp_pca = mean_squared_error(y_train, y_train_pred_ftcp_pca)
train_rmse_ftcp_pca = np.sqrt(train_mse_ftcp_pca)
train_mae_ftcp_pca = mean_absolute_error(y_train, y_train_pred_ftcp_pca)
train_r2_ftcp_pca = r2_score(y_train, y_train_pred_ftcp_pca)

results_ftcp_pca['train_metrics'] = {
    'r2': float(train_r2_ftcp_pca),
    'rmse': float(train_rmse_ftcp_pca),
    'mae': float(train_mae_ftcp_pca),
    'mse': float(train_mse_ftcp_pca)
}

print(f"   Train R²:   {train_r2_ftcp_pca:.6f}")
print(f"   Train RMSE: {train_rmse_ftcp_pca:.6f} eV/atom")
print(f"   Train MAE:  {train_mae_ftcp_pca:.6f} eV/atom")

# Evaluate on test set
print("\n📊 Evaluating on test set...")
sys.stdout.flush()
y_test_pred_ftcp_pca = model_ftcp_pca.predict(X_test_ftcp_pca)
test_mse_ftcp_pca = mean_squared_error(y_test, y_test_pred_ftcp_pca)
test_rmse_ftcp_pca = np.sqrt(test_mse_ftcp_pca)
test_mae_ftcp_pca = mean_absolute_error(y_test, y_test_pred_ftcp_pca)
test_r2_ftcp_pca = r2_score(y_test, y_test_pred_ftcp_pca)
test_evs_ftcp_pca = explained_variance_score(y_test, y_test_pred_ftcp_pca)

results_ftcp_pca['test_metrics'] = {
    'r2': float(test_r2_ftcp_pca),
    'rmse': float(test_rmse_ftcp_pca),
    'mae': float(test_mae_ftcp_pca),
    'mse': float(test_mse_ftcp_pca),
    'explained_variance': float(test_evs_ftcp_pca)
}

print(f"   Test R²:   {test_r2_ftcp_pca:.6f}")
print(f"   Test RMSE: {test_rmse_ftcp_pca:.6f} eV/atom")
print(f"   Test MAE:  {test_mae_ftcp_pca:.6f} eV/atom")

results_ftcp_pca['total_time'] = preprocessing_time_ftcp_pca + training_time_ftcp_pca

# ==============================================================================
# EXPERIMENT 2: SSL FEATURES (2,048D)
# ==============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: SSL FEATURES (2,048 dimensions)")
print("=" * 80)
sys.stdout.flush()

results_ssl = {}

print("\n🔧 Preprocessing SSL Features (already 2,048D)...")
print("   Applying StandardScaler...")
sys.stdout.flush()
scaler_ssl = StandardScaler()
start_time = time.time()
X_train_ssl_scaled = scaler_ssl.fit_transform(features_train)
X_test_ssl_scaled = scaler_ssl.transform(features_test)
preprocessing_time_ssl = time.time() - start_time
print(f"   ✅ Preprocessing completed in {preprocessing_time_ssl:.2f} seconds")
sys.stdout.flush()

results_ssl['preprocessing_time'] = preprocessing_time_ssl
results_ssl['input_dimensions'] = 2048
results_ssl['n_train_samples'] = len(X_train_ssl_scaled)
results_ssl['n_test_samples'] = len(X_test_ssl_scaled)

print("\n🤖 Training Linear Regression on SSL Features...")
print(f"   Training {len(X_train_ssl_scaled):,} samples with 2,048 features...")
print(f"   ⏰ Training started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()

model_ssl = LinearRegression(n_jobs=-1)
start_time = time.time()
model_ssl.fit(X_train_ssl_scaled, y_train)
training_time_ssl = time.time() - start_time

print(f"   ✅ TRAINING COMPLETED in {training_time_ssl:.2f} seconds")
print(f"   ⏰ Completed at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
sys.stdout.flush()
results_ssl['training_time'] = training_time_ssl

# Evaluate on train set
print("\n📈 Evaluating on training set...")
sys.stdout.flush()
y_train_pred_ssl = model_ssl.predict(X_train_ssl_scaled)
train_mse_ssl = mean_squared_error(y_train, y_train_pred_ssl)
train_rmse_ssl = np.sqrt(train_mse_ssl)
train_mae_ssl = mean_absolute_error(y_train, y_train_pred_ssl)
train_r2_ssl = r2_score(y_train, y_train_pred_ssl)

results_ssl['train_metrics'] = {
    'r2': float(train_r2_ssl),
    'rmse': float(train_rmse_ssl),
    'mae': float(train_mae_ssl),
    'mse': float(train_mse_ssl)
}

print(f"   Train R²:   {train_r2_ssl:.6f}")
print(f"   Train RMSE: {train_rmse_ssl:.6f} eV/atom")
print(f"   Train MAE:  {train_mae_ssl:.6f} eV/atom")

# Evaluate on test set
print("\n📊 Evaluating on test set...")
sys.stdout.flush()
y_test_pred_ssl = model_ssl.predict(X_test_ssl_scaled)
test_mse_ssl = mean_squared_error(y_test, y_test_pred_ssl)
test_rmse_ssl = np.sqrt(test_mse_ssl)
test_mae_ssl = mean_absolute_error(y_test, y_test_pred_ssl)
test_r2_ssl = r2_score(y_test, y_test_pred_ssl)
test_evs_ssl = explained_variance_score(y_test, y_test_pred_ssl)

results_ssl['test_metrics'] = {
    'r2': float(test_r2_ssl),
    'rmse': float(test_rmse_ssl),
    'mae': float(test_mae_ssl),
    'mse': float(test_mse_ssl),
    'explained_variance': float(test_evs_ssl)
}

print(f"   Test R²:   {test_r2_ssl:.6f}")
print(f"   Test RMSE: {test_rmse_ssl:.6f} eV/atom")
print(f"   Test MAE:  {test_mae_ssl:.6f} eV/atom")

results_ssl['total_time'] = preprocessing_time_ssl + training_time_ssl

# ==============================================================================
# COMPARISON AND ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("FAIR COMPARISON: FTCP+PCA vs SSL (Both 2,048D)")
print("=" * 80)
sys.stdout.flush()

comparison = {
    'experiment_info': {
        'task': 'Formation Energy Prediction',
        'model': 'Linear Regression',
        'split': '90% Train / 10% Test',
        'n_train': len(y_train),
        'n_test': len(y_test),
        'dimensions': 2048,
        'comparison_type': 'Fair (same dimensionality)',
        'date': pd.Timestamp.now().isoformat()
    },
    'ftcp_pca': results_ftcp_pca,
    'ssl': results_ssl
}

print("\n📊 TEST SET PERFORMANCE COMPARISON:")
print("-" * 80)
print(f"{'Metric':<25} {'FTCP+PCA':<20} {'SSL':<20} {'Winner':<15}")
print("-" * 80)

metrics = [
    ('R² Score', 'r2', True),
    ('RMSE (eV/atom)', 'rmse', False),
    ('MAE (eV/atom)', 'mae', False),
    ('Explained Variance', 'explained_variance', True)
]

for metric_name, metric_key, higher_is_better in metrics:
    ftcp_val = results_ftcp_pca['test_metrics'][metric_key]
    ssl_val = results_ssl['test_metrics'][metric_key]
    
    if higher_is_better:
        winner = "SSL" if ssl_val > ftcp_val else "FTCP+PCA"
    else:
        winner = "SSL" if ssl_val < ftcp_val else "FTCP+PCA"
    
    print(f"{metric_name:<25} {ftcp_val:<20.6f} {ssl_val:<20.6f} {winner:<15}")

print("\n⏱️  COMPUTATIONAL EFFICIENCY:")
print("-" * 80)
print(f"{'Stage':<25} {'FTCP+PCA':<20} {'SSL':<20} {'Difference':<15}")
print("-" * 80)
print(f"{'Preprocessing':<25} {preprocessing_time_ftcp_pca:<20.4f} {preprocessing_time_ssl:<20.4f} {preprocessing_time_ftcp_pca - preprocessing_time_ssl:>+6.2f}s")
print(f"{'Training':<25} {training_time_ftcp_pca:<20.4f} {training_time_ssl:<20.4f} {training_time_ftcp_pca - training_time_ssl:>+6.2f}s")
print(f"{'Total':<25} {results_ftcp_pca['total_time']:<20.4f} {results_ssl['total_time']:<20.4f} {results_ftcp_pca['total_time'] - results_ssl['total_time']:>+6.2f}s")

print("\n🔬 TRAIN-TEST GAP (Overfitting Analysis):")
print("-" * 80)
print(f"{'Model':<25} {'Train R²':<15} {'Test R²':<15} {'Gap':<15}")
print("-" * 80)
ftcp_gap = train_r2_ftcp_pca - test_r2_ftcp_pca
ssl_gap = train_r2_ssl - test_r2_ssl
print(f"{'FTCP+PCA':<25} {train_r2_ftcp_pca:<15.6f} {test_r2_ftcp_pca:<15.6f} {ftcp_gap:<15.6f}")
print(f"{'SSL':<25} {train_r2_ssl:<15.6f} {test_r2_ssl:<15.6f} {ssl_gap:<15.6f}")

# Determine winner
print("\n🏆 OVERALL WINNER:")
print("-" * 80)
if test_r2_ssl > test_r2_ftcp_pca:
    winner = "SSL"
    r2_diff = test_r2_ssl - test_r2_ftcp_pca
    r2_improvement = (r2_diff / abs(test_r2_ftcp_pca)) * 100 if test_r2_ftcp_pca > 0 else 100.0
    print(f"✅ SSL Features outperform FTCP+PCA")
    print(f"   SSL Test R²:       {test_r2_ssl:.6f}")
    print(f"   FTCP+PCA Test R²:  {test_r2_ftcp_pca:.6f}")
    print(f"   Absolute difference: +{r2_diff:.6f}")
    print(f"   Relative improvement: {r2_improvement:.2f}%")
else:
    winner = "FTCP+PCA"
    r2_diff = test_r2_ftcp_pca - test_r2_ssl
    r2_improvement = (r2_diff / abs(test_r2_ssl)) * 100 if test_r2_ssl > 0 else 100.0
    print(f"✅ FTCP+PCA outperforms SSL Features")
    print(f"   FTCP+PCA Test R²:  {test_r2_ftcp_pca:.6f}")
    print(f"   SSL Test R²:       {test_r2_ssl:.6f}")
    print(f"   Absolute difference: +{r2_diff:.6f}")
    print(f"   Relative improvement: {r2_improvement:.2f}%")

comparison['winner'] = {
    'model': winner,
    'r2_difference': float(abs(r2_diff)),
    'r2_improvement_percent': float(r2_improvement)
}

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)
sys.stdout.flush()

# Save JSON
json_path = results_dir / "pca_fair_comparison_90_10.json"
with open(json_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"\n✅ Saved results: {json_path}")

# Save predictions
np.save(results_dir / "ftcp_pca_test_predictions_90_10.npy", y_test_pred_ftcp_pca)
np.save(results_dir / "ssl_test_predictions_90_10.npy", y_test_pred_ssl)
np.save(results_dir / "test_true_labels_90_10.npy", y_test)
print(f"✅ Saved predictions")

# ==============================================================================
# CREATE VISUALIZATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)
sys.stdout.flush()

fig = plt.figure(figsize=(20, 12))

# 1. Predicted vs Actual - FTCP+PCA
ax1 = plt.subplot(2, 4, 1)
ax1.scatter(y_test, y_test_pred_ftcp_pca, alpha=0.5, s=10, c='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('True Formation Energy (eV/atom)', fontsize=12)
ax1.set_ylabel('Predicted (eV/atom)', fontsize=12)
ax1.set_title(f'FTCP+PCA (2,048D)\nTest R² = {test_r2_ftcp_pca:.4f}', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Predicted vs Actual - SSL
ax2 = plt.subplot(2, 4, 2)
ax2.scatter(y_test, y_test_pred_ssl, alpha=0.5, s=10, c='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('True Formation Energy (eV/atom)', fontsize=12)
ax2.set_ylabel('Predicted (eV/atom)', fontsize=12)
ax2.set_title(f'SSL Features (2,048D)\nTest R² = {test_r2_ssl:.4f}', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Performance Comparison
ax3 = plt.subplot(2, 4, 3)
metrics_names = ['R²', 'RMSE', 'MAE']
ftcp_vals = [test_r2_ftcp_pca, test_rmse_ftcp_pca, test_mae_ftcp_pca]
ssl_vals = [test_r2_ssl, test_rmse_ssl, test_mae_ssl]
x = np.arange(len(metrics_names))
width = 0.35
ax3.bar(x - width/2, ftcp_vals, width, label='FTCP+PCA', alpha=0.8, color='blue')
ax3.bar(x + width/2, ssl_vals, width, label='SSL', alpha=0.8, color='orange')
ax3.set_xlabel('Metrics', fontsize=12)
ax3.set_ylabel('Values', fontsize=12)
ax3.set_title('Test Set Performance', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics_names)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Time Comparison
ax4 = plt.subplot(2, 4, 4)
time_stages = ['Preprocessing', 'Training', 'Total']
ftcp_times = [preprocessing_time_ftcp_pca, training_time_ftcp_pca, results_ftcp_pca['total_time']]
ssl_times = [preprocessing_time_ssl, training_time_ssl, results_ssl['total_time']]
x = np.arange(len(time_stages))
ax4.bar(x - width/2, ftcp_times, width, label='FTCP+PCA', alpha=0.8, color='blue')
ax4.bar(x + width/2, ssl_times, width, label='SSL', alpha=0.8, color='orange')
ax4.set_xlabel('Stages', fontsize=12)
ax4.set_ylabel('Time (seconds)', fontsize=12)
ax4.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(time_stages)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Residuals - FTCP+PCA
ax5 = plt.subplot(2, 4, 5)
residuals_ftcp = y_test - y_test_pred_ftcp_pca
ax5.scatter(y_test_pred_ftcp_pca, residuals_ftcp, alpha=0.5, s=10, c='blue')
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicted (eV/atom)', fontsize=12)
ax5.set_ylabel('Residuals (eV/atom)', fontsize=12)
ax5.set_title(f'FTCP+PCA Residuals\nMAE = {test_mae_ftcp_pca:.4f}', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Residuals - SSL
ax6 = plt.subplot(2, 4, 6)
residuals_ssl = y_test - y_test_pred_ssl
ax6.scatter(y_test_pred_ssl, residuals_ssl, alpha=0.5, s=10, c='orange')
ax6.axhline(y=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Predicted (eV/atom)', fontsize=12)
ax6.set_ylabel('Residuals (eV/atom)', fontsize=12)
ax6.set_title(f'SSL Residuals\nMAE = {test_mae_ssl:.4f}', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. PCA Explained Variance
ax7 = plt.subplot(2, 4, 7)
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
ax7.plot(range(1, len(cumsum_var)+1), cumsum_var, 'b-', linewidth=2)
ax7.axhline(y=pca.explained_variance_ratio_.sum(), color='r', linestyle='--', lw=2, 
            label=f'Total: {pca.explained_variance_ratio_.sum():.4f}')
ax7.set_xlabel('Number of Components', fontsize=12)
ax7.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax7.set_title('PCA: Explained Variance', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Train-Test Gap
ax8 = plt.subplot(2, 4, 8)
models = ['FTCP+PCA', 'SSL']
train_r2s = [train_r2_ftcp_pca, train_r2_ssl]
test_r2s = [test_r2_ftcp_pca, test_r2_ssl]
gaps = [ftcp_gap, ssl_gap]
x = np.arange(len(models))
ax8.bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.8)
ax8.bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.8)
ax8.set_xlabel('Models', fontsize=12)
ax8.set_ylabel('R² Score', fontsize=12)
ax8.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(models)
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = results_dir / "pca_fair_comparison_90_10.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization: {plot_path}")
plt.close()

# ==============================================================================
# CREATE TEXT REPORT
# ==============================================================================

report = f"""
{'=' * 80}
FAIR COMPARISON: FTCP+PCA vs SSL FEATURES (Both 2,048D)
Formation Energy Prediction (90/10 Split)
{'=' * 80}

Experiment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Split: 90% Train ({len(y_train):,} samples) / 10% Test ({len(y_test):,} samples)
Model: Linear Regression (same for both)
Dimensionality: 2,048 (fair comparison)

{'=' * 80}
MOTIVATION FOR THIS COMPARISON
{'=' * 80}

Previous Comparison (Unfair):
✗ FTCP: 25,200 dimensions → Catastrophic failure (RMSE ~37,000 eV)
✓ SSL: 2,048 dimensions → Success (RMSE ~0.83 eV)
✗ Dimensionality mismatch makes comparison unfair

This Comparison (Fair):
✓ FTCP+PCA: 2,048 dimensions (linear dimensionality reduction)
✓ SSL: 2,048 dimensions (learned non-linear representations)
✓ Same dimensionality isolates representation quality

Key Question:
Is SSL's advantage due to:
(A) Dimensionality reduction alone (PCA can do this), OR
(B) Better learned representations (SSL's unique value)?

{'=' * 80}
DIMENSIONALITY REDUCTION DETAILS
{'=' * 80}

FTCP+PCA Pipeline:
1. StandardScaler: 25,200D → 25,200D (normalized)
2. PCA: 25,200D → 2,048D (linear projection)
3. Explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)
4. Information loss: {(1 - pca.explained_variance_ratio_.sum())*100:.2f}%

SSL Features:
1. Pre-trained: 25,200D → 2,048D (6-block encoder + bottleneck MLP)
2. Non-linear: Deep neural networks with multiple pretext tasks
3. Supervised by: 9 self-supervised tasks (no labels)
4. Information: Learned representations optimized for reconstruction

{'=' * 80}
RESULTS: TEST SET PERFORMANCE (90/10 Split)
{'=' * 80}

R² Score (Higher is Better):
- FTCP+PCA: {test_r2_ftcp_pca:.6f}
- SSL:      {test_r2_ssl:.6f}
- Winner:   {winner}
- Difference: {abs(test_r2_ssl - test_r2_ftcp_pca):.6f}

RMSE (Lower is Better):
- FTCP+PCA: {test_rmse_ftcp_pca:.6f} eV/atom
- SSL:      {test_rmse_ssl:.6f} eV/atom
- Winner:   {winner if test_rmse_ssl < test_rmse_ftcp_pca else 'FTCP+PCA'}
- Difference: {abs(test_rmse_ssl - test_rmse_ftcp_pca):.6f} eV/atom

MAE (Lower is Better):
- FTCP+PCA: {test_mae_ftcp_pca:.6f} eV/atom
- SSL:      {test_mae_ssl:.6f} eV/atom
- Winner:   {winner if test_mae_ssl < test_mae_ftcp_pca else 'FTCP+PCA'}

Explained Variance:
- FTCP+PCA: {test_evs_ftcp_pca:.6f}
- SSL:      {test_evs_ssl:.6f}

{'=' * 80}
TRAIN-TEST GAP ANALYSIS (Overfitting)
{'=' * 80}

FTCP+PCA:
- Train R²: {train_r2_ftcp_pca:.6f}
- Test R²:  {test_r2_ftcp_pca:.6f}
- Gap:      {ftcp_gap:.6f}
- Status:   {'⚠️ Overfitting' if ftcp_gap > 0.05 else '✅ Good'}

SSL:
- Train R²: {train_r2_ssl:.6f}
- Test R²:  {test_r2_ssl:.6f}
- Gap:      {ssl_gap:.6f}
- Status:   {'⚠️ Overfitting' if ssl_gap > 0.05 else '✅ Good'}

{'=' * 80}
COMPUTATIONAL EFFICIENCY
{'=' * 80}

Preprocessing Time:
- FTCP+PCA: {preprocessing_time_ftcp_pca:.2f}s (includes PCA fitting)
  - Scaling: {scaling_time:.2f}s
  - PCA: {pca_time:.2f}s
- SSL: {preprocessing_time_ssl:.2f}s (only scaling)
- Difference: {preprocessing_time_ftcp_pca - preprocessing_time_ssl:+.2f}s

Training Time:
- FTCP+PCA: {training_time_ftcp_pca:.2f}s
- SSL: {training_time_ssl:.2f}s
- Difference: {training_time_ftcp_pca - training_time_ssl:+.2f}s

Total Time:
- FTCP+PCA: {results_ftcp_pca['total_time']:.2f}s
- SSL: {results_ssl['total_time']:.2f}s
- Difference: {results_ftcp_pca['total_time'] - results_ssl['total_time']:+.2f}s

{'=' * 80}
CONCLUSION
{'=' * 80}

Winner: {winner}

Key Findings:
1. Fair comparison: Both models use exactly 2,048 dimensions
2. Performance: {'SSL shows better predictive accuracy' if winner == 'SSL' else 'FTCP+PCA shows better predictive accuracy'}
3. Interpretation: {'SSL learned representations outperform linear PCA' if winner == 'SSL' else 'PCA dimensionality reduction is sufficient'}
4. Efficiency: {'SSL is faster (no PCA needed)' if results_ssl['total_time'] < results_ftcp_pca['total_time'] else 'FTCP+PCA is faster'}

Scientific Interpretation:
{'SSL advantage comes from LEARNED REPRESENTATIONS, not just dimensionality reduction. Non-linear SSL encoders capture patterns that linear PCA cannot.' if winner == 'SSL' else 'PCA dimensionality reduction is sufficient for this task. SSL learned representations do not provide additional value over linear projection.'}

Comparison to Original (Unfair):
- Original FTCP (25,200D): RMSE ~37,000 eV (catastrophic)
- FTCP+PCA (2,048D): RMSE {test_rmse_ftcp_pca:.3f} eV (acceptable!)
- SSL (2,048D): RMSE {test_rmse_ssl:.3f} eV (best)

Dimensionality IS important, but representation quality ALSO matters!

Files Generated:
- pca_fair_comparison_90_10.json
- pca_fair_comparison_90_10.png
- *_90_10.npy (predictions)

{'=' * 80}
"""

report_path = results_dir / "PCA_FAIR_COMPARISON_REPORT_90_10.txt"
with open(report_path, 'w') as f:
    f.write(report)
print(f"✅ Saved detailed report: {report_path}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n⏰ End time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📁 All results saved to: {results_dir}")
print("\n✅ Ready for analysis!")
sys.stdout.flush()
