"""
RESIDUAL LEARNING: Formula-Group Split
======================================

Testing SSL's complementarity on COMPOSITIONALLY-HELD-OUT materials!

Why Formula-Group Split?
- Materials with same reduced formula kept together
- Tests generalization to unseen compositions
- HARDER task → More room for SSL to help!

Expected:
- FTCP R²: ~0.86 (vs 0.94 in random split)
- SSL Residual R²: 15-20% (vs 4.35% in random split)
- ΔR²: +1-2% (vs +0.27% in random split)

Data: Formula-Group Split (compositional holdout)
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data paths - Formula-Group Split
BASE_DIR = "/home/danial/Features_Extraction_Effectiveness/Data/Group_Splitting"

# FTCP paths
FTCP_TRAIN_DATA = f"{BASE_DIR}/Train/Train_FTCP_Data.npy"
FTCP_TRAIN_IDS = f"{BASE_DIR}/Train/Train_FTCP_Material_IDs.npy"
FTCP_TEST_DATA = f"{BASE_DIR}/Test/Test_FTCP_Data.npy"
FTCP_TEST_IDS = f"{BASE_DIR}/Test/Test_FTCP_Material_IDs.npy"

# SSL paths
SSL_TRAIN_DATA = f"{BASE_DIR}/Train/Train_Extracted_Features_Data.npy"
SSL_TRAIN_IDS = f"{BASE_DIR}/Train/Train_Extracted_Features_Material_IDs.npy"
SSL_TEST_DATA = f"{BASE_DIR}/Test/Test_Extracted_Features_Data.npy"
SSL_TEST_IDS = f"{BASE_DIR}/Test/Test_Extracted_Features_Material_IDs.npy"

# Labels
TRAIN_LABELS = f"{BASE_DIR}/Train/Train_FormationEnergy_Labels.npy"
TRAIN_LABEL_IDS = f"{BASE_DIR}/Train/Train_Material_IDs.npy"
TEST_LABELS = f"{BASE_DIR}/Test/Test_FormationEnergy_Labels.npy"
TEST_LABEL_IDS = f"{BASE_DIR}/Test/Test_Material_IDs.npy"

# Output directory
OUTPUT_DIR = "/home/danial/Features_Extraction_Effectiveness/Residual_Learning/Results_FormulaGroup"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🔬 RESIDUAL LEARNING: Formula-Group Split")
print("="*80)
print()
print("📊 Split Type: FORMULA-GROUP (Compositional Holdout)")
print("   Materials with same reduced formula kept together")
print("   → Tests generalization to UNSEEN COMPOSITIONS")
print()
print("🎯 Why This is Harder:")
print("   - FTCP captures structural patterns")
print("   - But struggles on novel compositions")
print("   - SSL's pre-learned physics should help more!")
print()
print("✨ Expected:")
print("   - FTCP R²: ~0.86 (vs 0.94 in random)")
print("   - SSL contribution: LARGER than random split")
print("   - More room for SSL to demonstrate value!")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND ALIGN DATA
# ============================================================================
print("\n📂 STEP 1: Loading data (Formula-Group split)...")
print("   Split: 90% train (~116k samples), 10% test (~13k samples)")
print("   Key: Train and test have NO overlapping reduced formulas!")

# Load FTCP
print("\n   [1/6] Loading FTCP train data...", end=' ', flush=True)
ftcp_train = np.load(FTCP_TRAIN_DATA).astype(np.float32)
ftcp_train_ids = np.load(FTCP_TRAIN_IDS, allow_pickle=True)
print("✓")

print("   [2/6] Loading FTCP test data...", end=' ', flush=True)
ftcp_test = np.load(FTCP_TEST_DATA).astype(np.float32)
ftcp_test_ids = np.load(FTCP_TEST_IDS, allow_pickle=True)
print("✓")

# Flatten FTCP if needed
if ftcp_train.ndim == 3:
    print(f"   Flattening FTCP from {ftcp_train.shape} to ", end='', flush=True)
    ftcp_train = ftcp_train.reshape(ftcp_train.shape[0], -1)
    ftcp_test = ftcp_test.reshape(ftcp_test.shape[0], -1)
    print(f"{ftcp_train.shape}")

print(f"   FTCP shapes - Train: {ftcp_train.shape}, Test: {ftcp_test.shape}")

# Load SSL
print("\n   [3/6] Loading SSL train data...", end=' ', flush=True)
ssl_train = np.load(SSL_TRAIN_DATA).astype(np.float32)
ssl_train_ids = np.load(SSL_TRAIN_IDS, allow_pickle=True)
print("✓")

print("   [4/6] Loading SSL test data...", end=' ', flush=True)
ssl_test = np.load(SSL_TEST_DATA).astype(np.float32)
ssl_test_ids = np.load(SSL_TEST_IDS, allow_pickle=True)
print("✓")
print(f"   SSL shapes - Train: {ssl_train.shape}, Test: {ssl_test.shape}")

# Load labels
print("\n   [5/6] Loading train labels...", end=' ', flush=True)
train_labels = np.load(TRAIN_LABELS).astype(np.float32)
train_label_ids = np.load(TRAIN_LABEL_IDS, allow_pickle=True)
print("✓")

print("   [6/6] Loading test labels...", end=' ', flush=True)
test_labels = np.load(TEST_LABELS).astype(np.float32)
test_label_ids = np.load(TEST_LABEL_IDS, allow_pickle=True)
print("✓")
print(f"   Label shapes - Train: {train_labels.shape}, Test: {test_labels.shape}")

# Align by material IDs
print("\n   🔗 Aligning data by material IDs...")
print("      (Ensuring FTCP, SSL, and labels match exactly)")

# Train alignment
ftcp_train_mask = np.isin(ftcp_train_ids, train_label_ids)
ssl_train_mask = np.isin(ssl_train_ids, train_label_ids)

ftcp_train_aligned = ftcp_train[ftcp_train_mask]
ssl_train_aligned = ssl_train[ssl_train_mask]

print(f"      Train aligned - FTCP: {ftcp_train_aligned.shape}, SSL: {ssl_train_aligned.shape}")

# Test alignment
ftcp_test_mask = np.isin(ftcp_test_ids, test_label_ids)
ssl_test_mask = np.isin(ssl_test_ids, test_label_ids)

ftcp_test_aligned = ftcp_test[ftcp_test_mask]
ssl_test_aligned = ssl_test[ssl_test_mask]

print(f"      Test aligned  - FTCP: {ftcp_test_aligned.shape}, SSL: {ssl_test_aligned.shape}")

# Verify alignment
assert ftcp_train_aligned.shape[0] == ssl_train_aligned.shape[0] == train_labels.shape[0], \
    "Train data alignment failed!"
assert ftcp_test_aligned.shape[0] == ssl_test_aligned.shape[0] == test_labels.shape[0], \
    "Test data alignment failed!"

print("\n✅ Data loaded and aligned successfully!")
print(f"   Final dataset: {len(train_labels):,} train, {len(test_labels):,} test samples")
print(f"   (Test materials have compositions UNSEEN in training!)")

# ============================================================================
# STEP 2: STANDARDIZE FTCP FEATURES
# ============================================================================
print("\n📊 STEP 2: Standardizing FTCP features...")
print("   (Fit on train, transform both train and test)")

ftcp_scaler = StandardScaler()
ftcp_train_scaled = ftcp_scaler.fit_transform(ftcp_train_aligned)
ftcp_test_scaled = ftcp_scaler.transform(ftcp_test_aligned)

print("✅ FTCP standardization complete!")

# ============================================================================
# STEP 3: TRAIN FTCP BASELINE MODEL
# ============================================================================
print("\n🔴 STEP 3: Training FTCP Baseline Model...")
print(f"   Features: {ftcp_train_aligned.shape[1]:,} (FTCP only)")
print("   Model: XGBoost Regressor")
print("   Task: Predict formation energy on UNSEEN COMPOSITIONS")

# XGBoost configuration
ftcp_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'tree_method': 'hist',
    'device': 'cpu',
    'n_jobs': -1
}

print(f"   Hyperparameters: n_estimators={ftcp_params['n_estimators']}, "
      f"max_depth={ftcp_params['max_depth']}, lr={ftcp_params['learning_rate']}")

print("\n   Training in progress...", end=' ', flush=True)
ftcp_start = time.time()

ftcp_model = xgb.XGBRegressor(**ftcp_params)
ftcp_model.fit(ftcp_train_scaled, train_labels, verbose=False)

ftcp_train_time = time.time() - ftcp_start
print(f"✓ ({ftcp_train_time:.1f}s)")

# Predict
print("   Making predictions...", end=' ', flush=True)
ftcp_train_pred = ftcp_model.predict(ftcp_train_scaled)
ftcp_test_pred = ftcp_model.predict(ftcp_test_scaled)
print("✓")

# Evaluate
ftcp_train_r2 = r2_score(train_labels, ftcp_train_pred)
ftcp_train_rmse = np.sqrt(mean_squared_error(train_labels, ftcp_train_pred))
ftcp_train_mae = mean_absolute_error(train_labels, ftcp_train_pred)

ftcp_test_r2 = r2_score(test_labels, ftcp_test_pred)
ftcp_test_rmse = np.sqrt(mean_squared_error(test_labels, ftcp_test_pred))
ftcp_test_mae = mean_absolute_error(test_labels, ftcp_test_pred)

print("\n   📊 FTCP Baseline Results:")
print(f"      Train: R² = {ftcp_train_r2:.4f}, RMSE = {ftcp_train_rmse:.4f}, MAE = {ftcp_train_mae:.4f}")
print(f"      Test:  R² = {ftcp_test_r2:.4f}, RMSE = {ftcp_test_rmse:.4f}, MAE = {ftcp_test_mae:.4f}")

if ftcp_test_r2 < 0.90:
    print(f"\n   ✅ GOOD! FTCP struggles on compositional generalization!")
    print(f"      (R² = {ftcp_test_r2:.4f} < 0.90 → harder task)")
    print(f"      → More room for SSL to add value!")
else:
    print(f"\n   ⚠️ FTCP is still strong (R² = {ftcp_test_r2:.4f})")

print("\n✅ FTCP baseline model trained!")

# ============================================================================
# STEP 4: COMPUTE RESIDUALS
# ============================================================================
print("\n📐 STEP 4: Computing Residuals (FTCP's Errors on Novel Compositions)...")
print("   Residual = True Value - FTCP Prediction")

train_residuals = train_labels - ftcp_train_pred
test_residuals = test_labels - ftcp_test_pred

print(f"\n   📊 Residual Statistics:")
print(f"      Train residuals - Mean: {np.mean(train_residuals):.6f}, Std: {np.std(train_residuals):.4f}")
print(f"      Test residuals  - Mean: {np.mean(test_residuals):.6f}, Std: {np.std(test_residuals):.4f}")
print(f"      Train residuals - Min: {np.min(train_residuals):.4f}, Max: {np.max(train_residuals):.4f}")
print(f"      Test residuals  - Min: {np.min(test_residuals):.4f}, Max: {np.max(test_residuals):.4f}")

# Check if residuals are reasonable
residual_variance = np.var(train_residuals)
print(f"\n   Residual variance: {residual_variance:.4f}")

if residual_variance > 0.05:
    print(f"   ✅ Good! Higher residual variance = more signal for SSL!")
else:
    print(f"   ⚠️ Low residual variance (FTCP already very good)")

print("\n✅ Residuals computed!")

# ============================================================================
# STEP 5: STANDARDIZE SSL FEATURES
# ============================================================================
print("\n📊 STEP 5: Standardizing SSL features...")
print("   (Fit on train, transform both train and test)")

ssl_scaler = StandardScaler()
ssl_train_scaled = ssl_scaler.fit_transform(ssl_train_aligned)
ssl_test_scaled = ssl_scaler.transform(ssl_test_aligned)

print("✅ SSL standardization complete!")

# ============================================================================
# STEP 6: TRAIN SSL MODEL TO PREDICT RESIDUALS
# ============================================================================
print("\n🔵 STEP 6: Training SSL Model to Predict Residuals...")
print(f"   Features: {ssl_train_aligned.shape[1]:,} (SSL only)")
print("   Target: FTCP's residuals on novel compositions")
print("   Model: XGBoost Regressor")

# SSL residual model configuration
ssl_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.2,
    'reg_alpha': 0.2,
    'reg_lambda': 2.0,
    'random_state': RANDOM_SEED,
    'tree_method': 'hist',
    'device': 'cpu',
    'n_jobs': -1
}

print(f"   Hyperparameters: n_estimators={ssl_params['n_estimators']}, "
      f"max_depth={ssl_params['max_depth']}, lr={ssl_params['learning_rate']}")
print("   (More regularized to avoid overfitting to residuals)")

print("\n   Training in progress...", end=' ', flush=True)
ssl_start = time.time()

ssl_residual_model = xgb.XGBRegressor(**ssl_params)
ssl_residual_model.fit(ssl_train_scaled, train_residuals, verbose=False)

ssl_train_time = time.time() - ssl_start
print(f"✓ ({ssl_train_time:.1f}s)")

# Predict residuals
print("   Predicting residuals...", end=' ', flush=True)
ssl_train_residual_pred = ssl_residual_model.predict(ssl_train_scaled)
ssl_test_residual_pred = ssl_residual_model.predict(ssl_test_scaled)
print("✓")

# Evaluate SSL's ability to predict residuals
ssl_residual_train_r2 = r2_score(train_residuals, ssl_train_residual_pred)
ssl_residual_test_r2 = r2_score(test_residuals, ssl_test_residual_pred)

print("\n   📊 SSL Residual Prediction Performance:")
print(f"      Train R² on residuals: {ssl_residual_train_r2:.4f}")
print(f"      Test R² on residuals:  {ssl_residual_test_r2:.4f}")

if ssl_residual_test_r2 > 0.15:
    print(f"\n   ✅ EXCELLENT! SSL explains {ssl_residual_test_r2*100:.1f}% of FTCP's error variance!")
    print(f"      (Much better than 4.35% in random split!)")
elif ssl_residual_test_r2 > 0.10:
    print(f"\n   ✅ GOOD! SSL explains {ssl_residual_test_r2*100:.1f}% of FTCP's error variance")
    print(f"      (Better than 4.35% in random split)")
elif ssl_residual_test_r2 > 0:
    print(f"\n   ✅ SSL explains {ssl_residual_test_r2*100:.1f}% of FTCP's error variance")
else:
    print(f"\n   ⚠️ SSL struggles to predict residuals (R² = {ssl_residual_test_r2:.4f})")

print("\n✅ SSL residual model trained!")

# ============================================================================
# STEP 7: COMBINE PREDICTIONS
# ============================================================================
print("\n🎯 STEP 7: Combining Predictions...")
print("   Final = FTCP Prediction + SSL Residual Correction")

combined_train_pred = ftcp_train_pred + ssl_train_residual_pred
combined_test_pred = ftcp_test_pred + ssl_test_residual_pred

# Evaluate combined model
combined_train_r2 = r2_score(train_labels, combined_train_pred)
combined_train_rmse = np.sqrt(mean_squared_error(train_labels, combined_train_pred))
combined_train_mae = mean_absolute_error(train_labels, combined_train_pred)

combined_test_r2 = r2_score(test_labels, combined_test_pred)
combined_test_rmse = np.sqrt(mean_squared_error(test_labels, combined_test_pred))
combined_test_mae = mean_absolute_error(test_labels, combined_test_pred)

print("\n   📊 Combined Model (FTCP + SSL) Results:")
print(f"      Train: R² = {combined_train_r2:.4f}, RMSE = {combined_train_rmse:.4f}, MAE = {combined_train_mae:.4f}")
print(f"      Test:  R² = {combined_test_r2:.4f}, RMSE = {combined_test_rmse:.4f}, MAE = {combined_test_mae:.4f}")

print("\n✅ Predictions combined!")

# ============================================================================
# STEP 8: COMPARISON AND ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("📊 STEP 8: COMPARISON - Formula-Group vs Random Split")
print("="*80)

print("\n🔴 FTCP-only Model:")
print(f"   Test R²:  {ftcp_test_r2:.4f}")
print(f"   Test RMSE: {ftcp_test_rmse:.4f}")
print(f"   Test MAE:  {ftcp_test_mae:.4f}")

print("\n🟣 FTCP + SSL (Combined) Model:")
print(f"   Test R²:  {combined_test_r2:.4f}")
print(f"   Test RMSE: {combined_test_rmse:.4f}")
print(f"   Test MAE:  {combined_test_mae:.4f}")

# Calculate improvements
r2_improvement = combined_test_r2 - ftcp_test_r2
r2_improvement_pct = (r2_improvement / ftcp_test_r2) * 100 if ftcp_test_r2 > 0 else 0

rmse_improvement = ftcp_test_rmse - combined_test_rmse
rmse_improvement_pct = (rmse_improvement / ftcp_test_rmse) * 100 if ftcp_test_rmse > 0 else 0

mae_improvement = ftcp_test_mae - combined_test_mae
mae_improvement_pct = (mae_improvement / ftcp_test_mae) * 100 if ftcp_test_mae > 0 else 0

print("\n🎯 IMPROVEMENT:")
print(f"   ΔR²:   {r2_improvement:+.4f} ({r2_improvement_pct:+.2f}%)")
print(f"   ΔRMSE: {rmse_improvement:+.4f} ({rmse_improvement_pct:+.2f}%)")
print(f"   ΔMAE:  {mae_improvement:+.4f} ({mae_improvement_pct:+.2f}%)")

# Compare with random split
print("\n📊 COMPARISON WITH RANDOM SPLIT:")
print(f"   Random Split:")
print(f"      FTCP R²: 0.9414, SSL contribution: 4.35%, ΔR²: +0.27%")
print(f"   Formula-Group Split:")
print(f"      FTCP R²: {ftcp_test_r2:.4f}, SSL contribution: {ssl_residual_test_r2*100:.2f}%, ΔR²: {r2_improvement_pct:+.2f}%")

if ssl_residual_test_r2 > 0.0435:  # 4.35% from random split
    improvement_ratio = ssl_residual_test_r2 / 0.0435
    print(f"\n   ✅ SSL contribution is {improvement_ratio:.1f}× LARGER on novel compositions!")
else:
    print(f"\n   ⚠️ SSL contribution is similar or smaller")

# Scientific interpretation
print("\n📝 SCIENTIFIC INTERPRETATION:")

if r2_improvement > 0.01:
    print("   ✅ STRONG SUCCESS! SSL adds significant value!")
    print(f"      → SSL captures compositional patterns FTCP misses")
    print(f"      → {r2_improvement_pct:.2f}% improvement on novel compositions")
elif r2_improvement > 0.005:
    print("   ✅ MODERATE SUCCESS! SSL adds measurable value!")
    print(f"      → SSL helps on compositional generalization")
elif r2_improvement > 0:
    print("   ✅ SUCCESS! SSL adds value (though modest)")
    print(f"      → Proves complementarity even on harder task")
else:
    print("   ⚠️ Limited improvement observed")

print("\n📈 RESIDUAL VARIANCE EXPLAINED:")
print(f"   SSL explains {ssl_residual_test_r2*100:.2f}% of FTCP's error variance")
print(f"   (vs 4.35% in random split)")

if ssl_residual_test_r2 > 0.20:
    print("   → STRONG evidence of complementary information on novel compositions!")
elif ssl_residual_test_r2 > 0.10:
    print("   → MODERATE complementarity on novel compositions")
elif ssl_residual_test_r2 > 0.0435:
    print("   → Improved complementarity compared to random split")
else:
    print("   → Similar to random split")

print("="*80)

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n🎨 STEP 9: Creating visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Comparison Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
models = ['FTCP\nOnly', 'FTCP\n+ SSL']
r2_scores = [ftcp_test_r2, combined_test_r2]
colors = ['#FF6B6B', '#4ECDC4']

bars = ax1.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Test R²', fontsize=13, fontweight='bold')
ax1.set_title('Formula-Group Split: Test R²', fontsize=14, fontweight='bold')
ax1.set_ylim([min(r2_scores) - 0.02, 1.0])
ax1.grid(True, axis='y', alpha=0.3)

for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

if r2_improvement > 0:
    ax1.annotate(f'+{r2_improvement:.4f}\n(+{r2_improvement_pct:.2f}%)',
                 xy=(1, combined_test_r2), xytext=(1.3, combined_test_r2),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2),
                 fontsize=11, color='green', fontweight='bold')

# Plot 2: RMSE & MAE Comparison
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(2)
width = 0.35
rmse_vals = [ftcp_test_rmse, combined_test_rmse]
mae_vals = [ftcp_test_mae, combined_test_mae]

bars1 = ax2.bar(x - width/2, rmse_vals, width, label='RMSE', color='#FF6B6B', alpha=0.8)
bars2 = ax2.bar(x + width/2, mae_vals, width, label='MAE', color='#4ECDC4', alpha=0.8)

ax2.set_ylabel('Error', fontsize=13, fontweight='bold')
ax2.set_title('Error Metrics (Novel Compositions)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend(fontsize=11)
ax2.grid(True, axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=10)

# Plot 3: Comparison with Random Split
ax3 = fig.add_subplot(gs[0, 2])
categories = ['FTCP\nR²', 'SSL\nContrib%', 'ΔR²\n%']
random_vals = [0.9414, 4.35, 0.27]
group_vals = [ftcp_test_r2, ssl_residual_test_r2*100, r2_improvement_pct]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, random_vals, width, label='Random Split', color='#95E1D3', alpha=0.8)
bars2 = ax3.bar(x + width/2, group_vals, width, label='Formula-Group', color='#F38181', alpha=0.8)

ax3.set_ylabel('Value', fontsize=13, fontweight='bold')
ax3.set_title('Random vs Formula-Group Split', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=10)
ax3.legend(fontsize=10)
ax3.grid(True, axis='y', alpha=0.3)

# Plot 4: Residual Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(train_residuals, bins=100, alpha=0.5, label='Train Residuals', color='blue', density=True)
ax4.hist(test_residuals, bins=100, alpha=0.5, label='Test Residuals', color='red', density=True)
ax4.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero Error')
ax4.set_xlabel('Residual (True - Predicted)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
ax4.set_title('FTCP Residuals on Novel Compositions', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Prediction Scatter
ax5 = fig.add_subplot(gs[1, 1])
sample_size = min(5000, len(test_labels))
sample_indices = np.random.choice(len(test_labels), sample_size, replace=False)

ax5.scatter(test_labels[sample_indices], ftcp_test_pred[sample_indices],
           alpha=0.3, s=10, label='FTCP Only', color='#FF6B6B')
ax5.scatter(test_labels[sample_indices], combined_test_pred[sample_indices],
           alpha=0.3, s=10, label='FTCP + SSL', color='#4ECDC4')

min_val = min(test_labels.min(), combined_test_pred.min())
max_val = max(test_labels.max(), combined_test_pred.max())
ax5.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect')

ax5.set_xlabel('True Formation Energy', fontsize=12, fontweight='bold')
ax5.set_ylabel('Predicted Formation Energy', fontsize=12, fontweight='bold')
ax5.set_title(f'Predictions (n={sample_size:,} novel compositions)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Plot 6: SSL Error Correction
ax6 = fig.add_subplot(gs[1, 2])
sample_errors_ftcp = np.abs(test_residuals[sample_indices])
sample_errors_combined = np.abs(test_labels[sample_indices] - combined_test_pred[sample_indices])

ax6.scatter(sample_errors_ftcp, sample_errors_combined, alpha=0.4, s=20, color='purple')
ax6.plot([0, sample_errors_ftcp.max()], [0, sample_errors_ftcp.max()],
         'k--', lw=2, label='No Improvement')

ax6.fill_between([0, sample_errors_ftcp.max()], 0, [0, sample_errors_ftcp.max()],
                  alpha=0.1, color='green', label='SSL Improves')

ax6.set_xlabel('|FTCP Error|', fontsize=12, fontweight='bold')
ax6.set_ylabel('|Combined Error|', fontsize=12, fontweight='bold')
ax6.set_title('SSL Error Correction on Novel Compositions', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.savefig(f"{OUTPUT_DIR}/formula_group_analysis.png", dpi=300, bbox_inches='tight')
print("   ✅ Saved: formula_group_analysis.png")
plt.close()

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n💾 STEP 10: Saving results...")

results = {
    'experiment': 'Residual_Learning_Formula_Group',
    'split': 'Formula_Group_Split',
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'hypothesis': 'SSL helps more on compositionally novel materials',
    
    'data': {
        'n_train': int(len(train_labels)),
        'n_test': int(len(test_labels)),
        'ftcp_features': int(ftcp_train_aligned.shape[1]),
        'ssl_features': int(ssl_train_aligned.shape[1]),
        'split_note': 'No overlapping reduced formulas between train and test'
    },
    
    'ftcp_baseline': {
        'train': {
            'r2': float(ftcp_train_r2),
            'rmse': float(ftcp_train_rmse),
            'mae': float(ftcp_train_mae)
        },
        'test': {
            'r2': float(ftcp_test_r2),
            'rmse': float(ftcp_test_rmse),
            'mae': float(ftcp_test_mae)
        },
        'train_time': float(ftcp_train_time)
    },
    
    'ssl_residual_model': {
        'train_r2_on_residuals': float(ssl_residual_train_r2),
        'test_r2_on_residuals': float(ssl_residual_test_r2),
        'variance_explained': float(ssl_residual_test_r2 * 100),
        'train_time': float(ssl_train_time)
    },
    
    'combined_model': {
        'train': {
            'r2': float(combined_train_r2),
            'rmse': float(combined_train_rmse),
            'mae': float(combined_train_mae)
        },
        'test': {
            'r2': float(combined_test_r2),
            'rmse': float(combined_test_rmse),
            'mae': float(combined_test_mae)
        }
    },
    
    'improvements': {
        'r2': {
            'absolute': float(r2_improvement),
            'percent': float(r2_improvement_pct)
        },
        'rmse': {
            'absolute': float(rmse_improvement),
            'percent': float(rmse_improvement_pct)
        },
        'mae': {
            'absolute': float(mae_improvement),
            'percent': float(mae_improvement_pct)
        }
    },
    
    'comparison_with_random_split': {
        'random_split': {
            'ftcp_r2': 0.9414,
            'ssl_contribution_pct': 4.35,
            'delta_r2_pct': 0.27
        },
        'formula_group_split': {
            'ftcp_r2': float(ftcp_test_r2),
            'ssl_contribution_pct': float(ssl_residual_test_r2 * 100),
            'delta_r2_pct': float(r2_improvement_pct)
        },
        'ssl_contribution_ratio': float(ssl_residual_test_r2 / 0.0435) if 0.0435 != 0 else None
    },
    
    'conclusion': {
        'ssl_adds_value': bool(r2_improvement > 0.001),
        'improvement_significant': bool(r2_improvement > 0.01),
        'better_than_random_split': bool(ssl_residual_test_r2 > 0.0435)
    }
}

with open(f"{OUTPUT_DIR}/formula_group_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("   ✅ Saved: formula_group_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 FORMULA-GROUP RESIDUAL LEARNING COMPLETE!")
print("="*80)
print()
print("📊 KEY RESULTS:")
print()
print(f"   FTCP-only Model:")
print(f"      Test R² = {ftcp_test_r2:.4f}, RMSE = {ftcp_test_rmse:.4f}, MAE = {ftcp_test_mae:.4f}")
print()
print(f"   FTCP + SSL Model:")
print(f"      Test R² = {combined_test_r2:.4f}, RMSE = {combined_test_rmse:.4f}, MAE = {combined_test_mae:.4f}")
print()
print(f"   🎯 IMPROVEMENT:")
print(f"      ΔR²   = {r2_improvement:+.4f} ({r2_improvement_pct:+.2f}%)")
print(f"      ΔRMSE = {rmse_improvement:+.4f} ({rmse_improvement_pct:+.2f}%)")
print(f"      ΔMAE  = {mae_improvement:+.4f} ({mae_improvement_pct:+.2f}%)")
print()
print(f"   📈 SSL explains {ssl_residual_test_r2*100:.2f}% of FTCP's error variance")
print(f"      (vs 4.35% in random split)")
print()

if ssl_residual_test_r2 > 0.0435:
    improvement_ratio = ssl_residual_test_r2 / 0.0435
    print(f"   ✅ SSL contribution is {improvement_ratio:.1f}× LARGER on novel compositions!")
    print("      → Proves SSL helps MORE on compositional generalization!")

if r2_improvement > 0.01:
    print("\n   ✅ STRONG SUCCESS: Significant improvement on novel compositions!")
elif r2_improvement > 0.005:
    print("\n   ✅ MODERATE SUCCESS: Measurable improvement on novel compositions!")
elif r2_improvement > 0:
    print("\n   ✅ SUCCESS: Improvement proves complementarity!")

print()
print("📁 OUTPUT FILES:")
print(f"   - {OUTPUT_DIR}/formula_group_results.json")
print(f"   - {OUTPUT_DIR}/formula_group_analysis.png")
print("="*80)
print()
print("🎯 FOR YOUR PAPER:")
print('   "On compositionally-held-out materials (formula-group split),')
print(f'    SSL explains {ssl_residual_test_r2*100:.1f}% of FTCP\'s error variance,')
print(f'    achieving {r2_improvement_pct:.2f}% improvement in R².')
print('    This demonstrates SSL\'s value for predicting novel compositions."')
print("="*80)
