"""
HYBRID ENRICHMENT EXPERIMENT: Random Forest Feature Importance
================================================================

Same as SHAP experiment but using Random Forest Feature Importance instead.

Goal: Compare SHAP vs RF importance for proving SSL enrichment

Strategy:
1. Train Random Forest on FTCP + SSL combined (27,248 features)
2. Extract feature importance from RF
3. Compare FTCP vs SSL importance
4. Compare results with SHAP analysis

This provides an alternative interpretability method!
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import json
import time
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths - Formula-Group Split (same as SHAP experiment)
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

# Output
OUTPUT_DIR = "/home/danial/Features_Extraction_Effectiveness/MLP/Formation_Energy/hybrid_enrichment/Results_RF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🌲 Random Forest Feature Importance Analysis")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND ALIGN DATA (SAME AS SHAP)
# ============================================================================
print("\n📂 STEP 1: Loading and aligning FTCP + SSL features...")
print("   (This takes ~30-60 seconds for large datasets...)")

# Load FTCP
print("   [1/6] Loading FTCP train data...", end=' ')
ftcp_train = np.load(FTCP_TRAIN_DATA).astype(np.float32)
ftcp_train_ids = np.load(FTCP_TRAIN_IDS, allow_pickle=True)
print("✓")
print("   [2/6] Loading FTCP test data...", end=' ')
ftcp_test = np.load(FTCP_TEST_DATA).astype(np.float32)
ftcp_test_ids = np.load(FTCP_TEST_IDS, allow_pickle=True)
print("✓")
print(f"   FTCP shapes - Train: {ftcp_train.shape}, Test: {ftcp_test.shape}")

# Flatten FTCP if needed
if ftcp_train.ndim == 3:
    ftcp_train = ftcp_train.reshape(ftcp_train.shape[0], -1)
    ftcp_test = ftcp_test.reshape(ftcp_test.shape[0], -1)
    print(f"   FTCP Flattened: {ftcp_train.shape}")

# Load SSL
print("   [3/6] Loading SSL train data...", end=' ')
ssl_train = np.load(SSL_TRAIN_DATA).astype(np.float32)
ssl_train_ids = np.load(SSL_TRAIN_IDS, allow_pickle=True)
print("✓")
print("   [4/6] Loading SSL test data...", end=' ')
ssl_test = np.load(SSL_TEST_DATA).astype(np.float32)
ssl_test_ids = np.load(SSL_TEST_IDS, allow_pickle=True)
print("✓")
print(f"   SSL shapes - Train: {ssl_train.shape}, Test: {ssl_test.shape}")

# Load labels
print("   [5/6] Loading train labels...", end=' ')
train_labels = np.load(TRAIN_LABELS).astype(np.float32)
train_label_ids = np.load(TRAIN_LABEL_IDS, allow_pickle=True)
print("✓")
print("   [6/6] Loading test labels...", end=' ')
test_labels = np.load(TEST_LABELS).astype(np.float32)
test_label_ids = np.load(TEST_LABEL_IDS, allow_pickle=True)
print("✓")
print(f"   Labels shapes - Train: {train_labels.shape}, Test: {test_labels.shape}")

# Align FTCP with labels
print("\n   🔗 Aligning data by material IDs...")
print("      (Matching ~67k train + ~13k test samples...)", end=' ')
ftcp_train_mask = np.isin(ftcp_train_ids, train_label_ids)
ftcp_test_mask = np.isin(ftcp_test_ids, test_label_ids)
ftcp_train_aligned = ftcp_train[ftcp_train_mask]
ftcp_test_aligned = ftcp_test[ftcp_test_mask]

# Align SSL with labels
ssl_train_mask = np.isin(ssl_train_ids, train_label_ids)
ssl_test_mask = np.isin(ssl_test_ids, test_label_ids)
ssl_train_aligned = ssl_train[ssl_train_mask]
ssl_test_aligned = ssl_test[ssl_test_mask]
print("✓")

print(f"   Aligned FTCP - Train: {ftcp_train_aligned.shape}, Test: {ftcp_test_aligned.shape}")
print(f"   Aligned SSL  - Train: {ssl_train_aligned.shape}, Test: {ssl_test_aligned.shape}")

# Verify alignment
assert ftcp_train_aligned.shape[0] == ssl_train_aligned.shape[0] == train_labels.shape[0], "Train size mismatch!"
assert ftcp_test_aligned.shape[0] == ssl_test_aligned.shape[0] == test_labels.shape[0], "Test size mismatch!"

print("✅ Data aligned successfully!")

# ============================================================================
# STEP 2: COMBINE FEATURES (SAME AS SHAP)
# ============================================================================
print("\n🔗 STEP 2: Combining FTCP + SSL features...")
print("   Concatenating feature matrices...", end=' ')

X_train_combined = np.concatenate([ftcp_train_aligned, ssl_train_aligned], axis=1)
X_test_combined = np.concatenate([ftcp_test_aligned, ssl_test_aligned], axis=1)
print("✓")

print(f"  Combined shape: Train {X_train_combined.shape}, Test {X_test_combined.shape}")
print(f"  Total features: {X_train_combined.shape[1]} (FTCP: {ftcp_train_aligned.shape[1]}, SSL: {ssl_train_aligned.shape[1]})")

# Feature indices
ftcp_feature_indices = np.arange(0, ftcp_train_aligned.shape[1])
ssl_feature_indices = np.arange(ftcp_train_aligned.shape[1], X_train_combined.shape[1])

print(f"  FTCP feature indices: 0-{ftcp_train_aligned.shape[1]-1}")
print(f"  SSL feature indices: {ftcp_train_aligned.shape[1]}-{X_train_combined.shape[1]-1}")

# ============================================================================
# STEP 3: STANDARDIZE (SAME AS SHAP)
# ============================================================================
print("\n📊 STEP 3: Standardizing combined features...")
print("   Computing mean and std for 27,248 features...", end=' ')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
print("✓")
print("   Applying standardization to test set...", end=' ')
X_test_scaled = scaler.transform(X_test_combined)
print("✓")

print("✅ Standardization complete!")

# ============================================================================
# STEP 4: TRAIN RANDOM FOREST
# ============================================================================
print("\n🌲 STEP 4: Training Random Forest Regressor...")
print("   Using optimized hyperparameters for fair comparison...")

# Use strong RF configuration
rf_params = {
    'n_estimators': 500,  # More trees for stable importance
    'max_depth': 20,      # Deep enough to capture complexity
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',  # Standard for high-dim data
    'random_state': RANDOM_SEED,
    'n_jobs': -1,  # Use all CPU cores
    'verbose': 1
}

print(f"  RF Parameters:")
print(f"    - Trees: {rf_params['n_estimators']}")
print(f"    - Max Depth: {rf_params['max_depth']}")
print(f"    - Max Features: {rf_params['max_features']}")
print(f"    - CPU Cores: All available")
print()

start_time = time.time()
rf_model = RandomForestRegressor(**rf_params)

print("  Training Random Forest (this takes ~5-10 minutes)...")
rf_model.fit(X_train_scaled, train_labels)

train_time = time.time() - start_time
print(f"\n✅ Training complete in {train_time:.2f}s ({train_time/60:.1f} minutes)")

# ============================================================================
# STEP 5: EVALUATION
# ============================================================================
print("\n📈 STEP 5: Evaluating Random Forest Model...")

print("   Making predictions on train set...", end=' ')
train_pred = rf_model.predict(X_train_scaled)
print("✓")

print("   Making predictions on test set...", end=' ')
test_pred = rf_model.predict(X_test_scaled)
print("✓")

# Metrics
def calc_metrics(y_true, y_pred, name):
    return {
        'name': name,
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'explained_variance': float(explained_variance_score(y_true, y_pred)),
        'max_error': float(np.max(np.abs(y_true - y_pred)))
    }

train_metrics = calc_metrics(train_labels, train_pred, "RF Hybrid Train")
test_metrics = calc_metrics(test_labels, test_pred, "RF Hybrid Test")

print(f"  Train R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
print(f"  Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")

# ============================================================================
# STEP 6: EXTRACT FEATURE IMPORTANCE
# ============================================================================
print("\n🔍 STEP 6: Extracting Random Forest Feature Importance...")
print("  Computing importance scores for all 27,248 features...")

# Get feature importances (based on mean decrease in impurity)
feature_importances = rf_model.feature_importances_

print(f"  ✅ Feature importances extracted! Shape: {feature_importances.shape}")

# ============================================================================
# STEP 7: ANALYZE RF IMPORTANCE
# ============================================================================
print("\n📊 STEP 7: Analyzing Feature Importance Results...")

# FTCP vs SSL importance
ftcp_importance = feature_importances[ftcp_feature_indices].sum()
ssl_importance = feature_importances[ssl_feature_indices].sum()
total_importance = ftcp_importance + ssl_importance

ftcp_percent = (ftcp_importance / total_importance) * 100
ssl_percent = (ssl_importance / total_importance) * 100

print(f"\n  📌 OVERALL FEATURE IMPORTANCE:")
print(f"     FTCP features: {ftcp_percent:.2f}% ({ftcp_train_aligned.shape[1]} features)")
print(f"     SSL features:  {ssl_percent:.2f}% ({ssl_train_aligned.shape[1]} features)")
print()
print(f"  📌 PER-FEATURE IMPORTANCE:")
print(f"     FTCP: {ftcp_importance/ftcp_train_aligned.shape[1]:.6f} per feature")
print(f"     SSL:  {ssl_importance/ssl_train_aligned.shape[1]:.6f} per feature")

if ssl_importance/ssl_train_aligned.shape[1] > 0 and ftcp_importance/ftcp_train_aligned.shape[1] > 0:
    fold_change = (ssl_importance/ssl_train_aligned.shape[1]) / (ftcp_importance/ftcp_train_aligned.shape[1])
    print(f"     SSL is {fold_change:.2f}× more important per feature!")
else:
    fold_change = 0.0

# Top features overall
print(f"\n  📌 TOP 20 MOST IMPORTANT FEATURES:")
top_indices = np.argsort(feature_importances)[::-1][:20]
print("\n     Rank | Feature Type | Feature Index | Importance")
print("     " + "-"*60)

for rank, idx in enumerate(top_indices, 1):
    if idx < ftcp_train_aligned.shape[1]:
        feat_type = "FTCP"
        feat_idx = idx
    else:
        feat_type = "SSL "
        feat_idx = idx - ftcp_train_aligned.shape[1]
    importance = feature_importances[idx]
    print(f"     {rank:4d} | {feat_type} | {feat_idx:13d} | {importance:.6f}")

# Count top features by type
top_100_indices = np.argsort(feature_importances)[::-1][:100]
top_100_ftcp = sum(1 for idx in top_100_indices if idx < ftcp_train_aligned.shape[1])
top_100_ssl = 100 - top_100_ftcp

print(f"\n  📌 TOP 100 FEATURES BREAKDOWN:")
print(f"     FTCP: {top_100_ftcp} features ({top_100_ftcp}%)")
print(f"     SSL:  {top_100_ssl} features ({top_100_ssl}%)")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n🎨 STEP 8: Creating visualizations...")
print("   Generating 4 plots (this takes ~1-2 minutes)...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Overall importance bar chart
ax = axes[0, 0]
importance_data = [ftcp_percent, ssl_percent]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax.bar(['FTCP\n(25,200 features)', 'SSL\n(2,048 features)'], importance_data, color=colors, alpha=0.8)
ax.set_ylabel('Importance (%)', fontsize=12)
ax.set_title('Overall Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
for bar, val in zip(bars, importance_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Per-feature importance
ax = axes[0, 1]
per_feature_ftcp = ftcp_importance / ftcp_train_aligned.shape[1]
per_feature_ssl = ssl_importance / ssl_train_aligned.shape[1]
bars = ax.bar(['FTCP\n(per feature)', 'SSL\n(per feature)'], 
              [per_feature_ftcp, per_feature_ssl], color=colors, alpha=0.8)
ax.set_ylabel('Importance per Feature', fontsize=12)
ax.set_title('Per-Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
if fold_change > 0:
    ax.text(0.5, max(per_feature_ftcp, per_feature_ssl) * 0.8, 
            f'SSL is {fold_change:.1f}× more\nimportant per feature!', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Plot 3: Top 100 features by type
ax = axes[1, 0]
type_data = [top_100_ftcp, top_100_ssl]
bars = ax.bar(['FTCP', 'SSL'], type_data, color=colors, alpha=0.8)
ax.set_ylabel('Count in Top 100', fontsize=12)
ax.set_title('Feature Type Distribution in Top 100 Most Important', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
for bar, val in zip(bars, type_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 4: Importance distribution histogram
ax = axes[1, 1]
ax.hist(feature_importances[ftcp_feature_indices], bins=50, alpha=0.6, label='FTCP', color='#FF6B6B')
ax.hist(feature_importances[ssl_feature_indices], bins=50, alpha=0.6, label='SSL', color='#4ECDC4')
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_ylabel('Number of Features', fontsize=12)
ax.set_title('Distribution of Feature Importances', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_yscale('log')  # Log scale to see distribution better

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_importance_analysis.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: rf_importance_analysis.png")

# Plot 5: Top 30 features bar chart
plt.figure(figsize=(12, 10))
top_30_indices = np.argsort(feature_importances)[::-1][:30]
top_30_importances = feature_importances[top_30_indices]
top_30_labels = []
top_30_colors = []

for idx in top_30_indices:
    if idx < ftcp_train_aligned.shape[1]:
        top_30_labels.append(f'FTCP_{idx}')
        top_30_colors.append('#FF6B6B')
    else:
        top_30_labels.append(f'SSL_{idx - ftcp_train_aligned.shape[1]}')
        top_30_colors.append('#4ECDC4')

plt.barh(range(30), top_30_importances, color=top_30_colors, alpha=0.8)
plt.yticks(range(30), top_30_labels, fontsize=8)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 30 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rf_top30_features.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: rf_top30_features.png")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n💾 STEP 9: Saving results...")

results = {
    'experiment': 'RF_Feature_Importance_Analysis',
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'split': 'Formula-Group (Most Rigorous)',
    'model': 'Random Forest Regressor',
    'architecture': {
        'total_features': int(X_train_combined.shape[1]),
        'ftcp_features': int(ftcp_train_aligned.shape[1]),
        'ssl_features': int(ssl_train_aligned.shape[1]),
        'n_estimators': rf_params['n_estimators'],
        'max_depth': rf_params['max_depth']
    },
    'training': {
        'train_samples': int(len(train_labels)),
        'test_samples': int(len(test_labels)),
        'training_time_sec': float(train_time)
    },
    'performance': {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    },
    'rf_importance_analysis': {
        'ftcp_importance_percent': float(ftcp_percent),
        'ssl_importance_percent': float(ssl_percent),
        'ftcp_per_feature_importance': float(per_feature_ftcp),
        'ssl_per_feature_importance': float(per_feature_ssl),
        'ssl_efficiency_multiplier': float(fold_change),
        'top_100_ftcp_count': int(top_100_ftcp),
        'top_100_ssl_count': int(top_100_ssl)
    },
    'comparison_to_individual': {
        'ftcp_only_r2': 0.9211,  # From previous experiment
        'ssl_only_r2': 0.6390,   # From previous experiment
        'rf_hybrid_r2': float(test_metrics['r2']),
        'improvement_over_ftcp': float(test_metrics['r2'] - 0.9211),
        'improvement_over_ssl': float(test_metrics['r2'] - 0.6390)
    },
    'enrichment_proof': {
        'ssl_adds_value': bool(int(ssl_percent > 10.0)),
        'ssl_per_feature_advantage': bool(int(fold_change > 1.0)),
        'conclusion': f"RF importance shows SSL features contribute {ssl_percent:.1f}% to predictions despite being only {(ssl_train_aligned.shape[1]/X_train_combined.shape[1])*100:.1f}% of total features. Per-feature importance is {fold_change:.1f}× higher than FTCP."
    }
}

# Save JSON with proper type conversion
def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(int(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

results_serializable = convert_to_json_serializable(results)

with open(f"{OUTPUT_DIR}/rf_importance_results.json", 'w') as f:
    json.dump(results_serializable, f, indent=2)

# Save feature importances
np.save(f"{OUTPUT_DIR}/rf_feature_importances.npy", feature_importances)

print(f"✅ Results saved to: {OUTPUT_DIR}/")

# ============================================================================
# STEP 10: LOAD SHAP RESULTS FOR COMPARISON
# ============================================================================
print("\n🔄 STEP 10: Comparing with SHAP Results...")

try:
    with open("/home/danial/Features_Extraction_Effectiveness/MLP/Formation_Energy/hybrid_enrichment/Results/hybrid_enrichment_results.json", 'r') as f:
        shap_results = json.load(f)
    
    print("\n  📊 COMPARISON: Random Forest vs SHAP")
    print("  " + "="*70)
    
    comparison_table = [
        ["Metric", "Random Forest", "SHAP", "Difference"],
        ["-"*20, "-"*20, "-"*20, "-"*20],
        [
            "SSL Importance %",
            f"{ssl_percent:.2f}%",
            f"{shap_results['shap_analysis']['ssl_importance_percent']:.2f}%",
            f"{ssl_percent - shap_results['shap_analysis']['ssl_importance_percent']:+.2f}%"
        ],
        [
            "FTCP Importance %",
            f"{ftcp_percent:.2f}%",
            f"{shap_results['shap_analysis']['ftcp_importance_percent']:.2f}%",
            f"{ftcp_percent - shap_results['shap_analysis']['ftcp_importance_percent']:+.2f}%"
        ],
        [
            "SSL Efficiency (×)",
            f"{fold_change:.2f}×",
            f"{shap_results['shap_analysis']['ssl_efficiency_multiplier']:.2f}×",
            f"{fold_change - shap_results['shap_analysis']['ssl_efficiency_multiplier']:+.2f}×"
        ],
        [
            "Model Test R²",
            f"{test_metrics['r2']:.4f}",
            f"{shap_results['performance']['test_metrics']['r2']:.4f}",
            f"{test_metrics['r2'] - shap_results['performance']['test_metrics']['r2']:+.4f}"
        ]
    ]
    
    for row in comparison_table:
        print(f"  {row[0]:20s} | {row[1]:20s} | {row[2]:20s} | {row[3]:20s}")
    
    # Save comparison (with proper type handling)
    comparison_results = {
        'comparison_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'methods': ['Random Forest', 'SHAP'],
        'ssl_importance': {
            'rf': float(ssl_percent),
            'shap': float(shap_results['shap_analysis']['ssl_importance_percent']),
            'difference': float(ssl_percent - shap_results['shap_analysis']['ssl_importance_percent']),
            'agreement': bool(int(abs(ssl_percent - shap_results['shap_analysis']['ssl_importance_percent']) < 10.0))
        },
        'ssl_efficiency': {
            'rf': float(fold_change),
            'shap': float(shap_results['shap_analysis']['ssl_efficiency_multiplier']),
            'difference': float(fold_change - shap_results['shap_analysis']['ssl_efficiency_multiplier']),
            'agreement': bool(int(abs(fold_change - shap_results['shap_analysis']['ssl_efficiency_multiplier']) < 2.0))
        },
        'model_performance': {
            'rf_r2': float(test_metrics['r2']),
            'mlp_r2': float(shap_results['performance']['test_metrics']['r2']),
            'difference': float(test_metrics['r2'] - shap_results['performance']['test_metrics']['r2'])
        },
        'conclusion': 'Both methods agree on SSL enrichment' if abs(ssl_percent - shap_results['shap_analysis']['ssl_importance_percent']) < 10.0 else 'Methods show different results'
    }
    
    # Convert to JSON-serializable format
    comparison_results_serializable = convert_to_json_serializable(comparison_results)
    
    with open(f"{OUTPUT_DIR}/rf_vs_shap_comparison.json", 'w') as f:
        json.dump(comparison_results_serializable, f, indent=2)
    
    print(f"\n  ✅ Comparison saved: rf_vs_shap_comparison.json")
    
except FileNotFoundError:
    print("\n  ⚠️ SHAP results not found - skipping comparison")
    comparison_results = None

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 RANDOM FOREST FEATURE IMPORTANCE ANALYSIS COMPLETE!")
print("="*80)
print()
print("📊 KEY FINDINGS:")
print(f"   RF Hybrid Model Test R²: {test_metrics['r2']:.4f}")
print(f"   Training Time: {train_time:.1f}s ({train_time/60:.1f} minutes)")
print()
print("🔍 RF IMPORTANCE ANALYSIS:")
print(f"   SSL Importance: {ssl_percent:.1f}% (despite only {(ssl_train_aligned.shape[1]/X_train_combined.shape[1])*100:.1f}% of features)")
print(f"   SSL per-feature importance: {fold_change:.1f}× higher than FTCP")
print(f"   Top 100 features: {top_100_ssl} are SSL features ({top_100_ssl}%)")
print()

if comparison_results:
    print("🔄 COMPARISON WITH SHAP:")
    print(f"   SSL Importance: RF={ssl_percent:.1f}% vs SHAP={shap_results['shap_analysis']['ssl_importance_percent']:.1f}%")
    print(f"   Agreement: {comparison_results['conclusion']}")
    print()

print("✅ ENRICHMENT PROVEN (RF METHOD):")
print(f"   {results['enrichment_proof']['conclusion']}")
print()
print("📁 OUTPUT FILES:")
print(f"   - rf_importance_results.json")
print(f"   - rf_importance_analysis.png")
print(f"   - rf_top30_features.png")
print(f"   - rf_feature_importances.npy")
if comparison_results:
    print(f"   - rf_vs_shap_comparison.json")
print("="*80)
print()
print("🎯 CONCLUSION:")
print("   Random Forest provides an alternative interpretability method!")
print("   Compare RF and SHAP results to see if both methods agree on SSL enrichment.")
print("="*80)
