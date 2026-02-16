#!/usr/bin/env python3
"""
STEP 2 - BLOCK 1: ELEMENT MATRIX TRAINING
==========================================
Self-supervised learning on element composition matrix (FTCP rows 0-102, columns 0-3)

Architecture: Dual-Stream MLP
- Separate processing of content (raw values) and sparsity patterns (binary mask)
- Multi-task SSL: Element prediction, Position prediction, Count prediction

Input:  103×4 binary element presence matrix (97% sparse)
Output: 64-dimensional dense feature representation

Author: Materials Informatics Research
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

print("\n" + "="*80)
print("BLOCK 1: ELEMENT MATRIX - DUAL-STREAM MLP WITH MULTI-TASK SSL")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading FTCP data...")

X_train = np.load('X_train_final.npy', mmap_mode='r')
X_val = np.load('X_val_final.npy', mmap_mode='r')
X_test = np.load('X_test_final.npy', mmap_mode='r')

y_train = np.load('y_train_final.npy')
y_val = np.load('y_val_final.npy')
y_test = np.load('y_test_final.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Extract Block 1: Element matrix (rows 0-102, columns 0-3)
X_elem_train = X_train[:, 0:103, 0:4].copy()
X_elem_val = X_val[:, 0:103, 0:4].copy()
X_elem_test = X_test[:, 0:103, 0:4].copy()

print(f"\nBlock 1 shapes: Train={X_elem_train.shape}, Val={X_elem_val.shape}, Test={X_elem_test.shape}")

# Analyze sparsity
for name, X_elem in [("TRAIN", X_elem_train), ("VAL", X_elem_val), ("TEST", X_elem_test)]:
    flat = X_elem.reshape(X_elem.shape[0], -1)
    sparsity = (1 - (flat > 0).mean()) * 100
    elements_per_sample = (flat > 0).sum(axis=1)
    print(f"  {name}: {sparsity:.1f}% sparse, {elements_per_sample.mean():.1f}±{elements_per_sample.std():.1f} non-zero elements/sample")

# ============================================================================
# 2. DATASET CLASS
# ============================================================================
print("\n[2/6] Creating datasets...")

class Block1ElementDataset(Dataset):
    """
    Dataset for Block 1 element matrix with self-supervised learning tasks.
    
    SSL Tasks:
    1. Masked Element Modeling (MEM): Predict which element was masked
    2. Masked Position Prediction: Predict position (0-3) of masked element
    3. Element Count Prediction: Classify number of distinct elements (3 classes)
    """
    
    def __init__(self, X_elem, y_synth, seed=42):
        self.X_elem = X_elem  # (N, 103, 4) numpy array
        self.y_synth = y_synth  # (N,) numpy array
        self.seed = seed
        
    def __len__(self):
        return len(self.X_elem)
    
    def __getitem__(self, idx):
        elem_matrix = self.X_elem[idx]  # (103, 4)
        
        # Create binary presence matrix
        elem_binary = (elem_matrix > 0).astype(np.float32)
        
        # Find all non-zero positions
        non_zero_positions = np.argwhere(elem_binary > 0)
        
        # Initialize targets
        target_element = -1
        target_pos = -1
        masked_matrix = elem_binary.copy()
        
        # Masking: randomly mask ONE non-zero position
        if len(non_zero_positions) > 0:
            # Use deterministic random based on idx for reproducibility
            rng = np.random.default_rng(self.seed + idx)
            mask_idx = rng.integers(0, len(non_zero_positions))
            pos = non_zero_positions[mask_idx]
            
            target_element = int(pos[0])  # Element index (0-102)
            target_pos = int(pos[1])      # Position index (0-3)
            
            # Mask this specific position
            masked_matrix[target_element, target_pos] = 0.0
        
        # Count distinct elements (for classification task)
        element_count = int((elem_binary.sum(axis=1) > 0).sum())
        
        # Convert to 3-class problem: 2 elements → class 0, 3 → class 1, 4+ → class 2
        count_class = min(max(element_count - 2, 0), 2)
        
        return {
            'masked': torch.from_numpy(masked_matrix),
            'target_element': torch.tensor(target_element, dtype=torch.long),
            'target_pos': torch.tensor(target_pos, dtype=torch.long),
            'count_class': torch.tensor(count_class, dtype=torch.long),
            'synthesis_label': torch.tensor(self.y_synth[idx], dtype=torch.long)
        }

# Create datasets
train_dataset = Block1ElementDataset(X_elem_train, y_train)
val_dataset = Block1ElementDataset(X_elem_val, y_val)
test_dataset = Block1ElementDataset(X_elem_test, y_test)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================
print("\n[3/6] Building Dual-Stream MLP model...")

class Block1ElementMatrixNet(nn.Module):
    """
    Dual-Stream MLP for Element Matrix Feature Extraction.
    
    Architecture:
    - Stream 1: Content projection (raw element matrix values)
    - Stream 2: Sparsity projection (binary presence mask)
    - Fusion: Concatenate both streams
    - Processing: 3 fully-connected layers with BatchNorm and Dropout
    - Heads: 3 task-specific prediction heads
    """
    
    def __init__(self, input_dim=103*4, hidden_dim=512, output_dim=64):
        super(Block1ElementMatrixNet, self).__init__()
        
        # Dual-stream input processing
        self.content_proj = nn.Linear(input_dim, hidden_dim)
        self.sparsity_proj = nn.Linear(input_dim, hidden_dim // 2)
        
        # Combined processing layers
        self.combine = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # Task-specific heads for self-supervision
        self.element_head = nn.Linear(output_dim, 103)  # Predict which element (103 classes)
        self.position_head = nn.Linear(output_dim, 4)   # Predict position (4 classes)
        self.count_head = nn.Linear(output_dim, 3)      # Predict element count (3 classes)
    
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # (B, 412)
        
        # Dual-stream processing
        content_feat = self.content_proj(x_flat)
        sparsity_mask = (x_flat > 0).float()
        sparsity_feat = self.sparsity_proj(sparsity_mask)
        
        # Combine streams
        combined = torch.cat([content_feat, sparsity_feat], dim=1)
        h1 = F.relu(self.bn1(self.combine(combined)))
        h1 = self.dropout1(h1)
        
        # Hidden processing
        h2 = F.relu(self.bn2(self.hidden(h1)))
        h2 = self.dropout2(h2)
        
        # Output features
        features = self.bn3(self.output(h2))
        
        if return_features:
            return features
        
        # Task predictions
        return {
            'features': features,
            'element_pred': self.element_head(features),
            'position_pred': self.position_head(features),
            'count_pred': self.count_head(features)
        }

# Initialize model
model = Block1ElementMatrixNet(output_dim=64)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4. TRAINING SETUP
# ============================================================================
print("\n[4/6] Setting up training...")

# Hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10

# Loss weights for multi-task learning
WEIGHT_ELEMENT = 0.4
WEIGHT_POSITION = 0.2
WEIGHT_COUNT = 0.4

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss weights: Element={WEIGHT_ELEMENT}, Position={WEIGHT_POSITION}, Count={WEIGHT_COUNT}")

# ============================================================================
# 5. TRAINING AND VALIDATION FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    element_correct = 0
    position_correct = 0
    count_correct = 0
    valid_samples = 0
    total_samples = 0
    
    for batch in dataloader:
        masked = batch['masked'].to(device)
        target_element = batch['target_element'].to(device)
        target_pos = batch['target_pos'].to(device)
        count_class = batch['count_class'].to(device)
        
        optimizer.zero_grad()
        outputs = model(masked)
        
        # Only compute element/position loss for valid targets (not -1)
        valid_mask = (target_element >= 0) & (target_pos >= 0)
        
        element_loss = torch.tensor(0.0, device=device)
        position_loss = torch.tensor(0.0, device=device)
        
        if valid_mask.sum() > 0:
            element_loss = F.cross_entropy(
                outputs['element_pred'][valid_mask], 
                target_element[valid_mask]
            )
            position_loss = F.cross_entropy(
                outputs['position_pred'][valid_mask], 
                target_pos[valid_mask]
            )
            
            # Accuracy calculation
            element_pred = outputs['element_pred'][valid_mask].argmax(dim=1)
            position_pred = outputs['position_pred'][valid_mask].argmax(dim=1)
            element_correct += (element_pred == target_element[valid_mask]).sum().item()
            position_correct += (position_pred == target_pos[valid_mask]).sum().item()
            valid_samples += valid_mask.sum().item()
        
        # Count loss (always computed)
        count_loss = F.cross_entropy(outputs['count_pred'], count_class)
        count_pred = outputs['count_pred'].argmax(dim=1)
        count_correct += (count_pred == count_class).sum().item()
        total_samples += len(count_class)
        
        # Combined weighted loss
        loss = WEIGHT_ELEMENT * element_loss + WEIGHT_POSITION * position_loss + WEIGHT_COUNT * count_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'element_acc': element_correct / max(valid_samples, 1),
        'position_acc': position_correct / max(valid_samples, 1),
        'count_acc': count_correct / total_samples
    }


@torch.no_grad()
def validate_model(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    element_correct = 0
    position_correct = 0
    count_correct = 0
    valid_samples = 0
    total_samples = 0
    
    for batch in dataloader:
        masked = batch['masked'].to(device)
        target_element = batch['target_element'].to(device)
        target_pos = batch['target_pos'].to(device)
        count_class = batch['count_class'].to(device)
        
        outputs = model(masked)
        
        valid_mask = (target_element >= 0) & (target_pos >= 0)
        
        element_loss = torch.tensor(0.0, device=device)
        position_loss = torch.tensor(0.0, device=device)
        
        if valid_mask.sum() > 0:
            element_loss = F.cross_entropy(
                outputs['element_pred'][valid_mask], 
                target_element[valid_mask]
            )
            position_loss = F.cross_entropy(
                outputs['position_pred'][valid_mask], 
                target_pos[valid_mask]
            )
            
            element_pred = outputs['element_pred'][valid_mask].argmax(dim=1)
            position_pred = outputs['position_pred'][valid_mask].argmax(dim=1)
            element_correct += (element_pred == target_element[valid_mask]).sum().item()
            position_correct += (position_pred == target_pos[valid_mask]).sum().item()
            valid_samples += valid_mask.sum().item()
        
        count_loss = F.cross_entropy(outputs['count_pred'], count_class)
        count_pred = outputs['count_pred'].argmax(dim=1)
        count_correct += (count_pred == count_class).sum().item()
        total_samples += len(count_class)
        
        loss = WEIGHT_ELEMENT * element_loss + WEIGHT_POSITION * position_loss + WEIGHT_COUNT * count_loss
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'element_acc': element_correct / max(valid_samples, 1),
        'position_acc': position_correct / max(valid_samples, 1),
        'count_acc': count_correct / total_samples
    }

# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
print("\n[5/6] Training...")

best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_position_acc': [], 'val_count_acc': []}

for epoch in range(NUM_EPOCHS):
    train_metrics = train_epoch(model, train_loader, optimizer, device)
    val_metrics = validate_model(model, val_loader, device)
    scheduler.step()
    
    # Save history
    history['train_loss'].append(train_metrics['loss'])
    history['val_loss'].append(val_metrics['loss'])
    history['val_position_acc'].append(val_metrics['position_acc'])
    history['val_count_acc'].append(val_metrics['count_acc'])
    
    # Early stopping
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        torch.save(model.state_dict(), 'block1_best.pth')
        best_epoch = epoch + 1
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch < 3:
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Loss: Train={train_metrics['loss']:.4f} Val={val_metrics['loss']:.4f} | "
              f"Pos Acc: {val_metrics['position_acc']:.4f} | "
              f"Count Acc: {val_metrics['count_acc']:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (best: epoch {best_epoch})")
        break

print(f"\nTraining completed! Best model saved at epoch {best_epoch}")

# ============================================================================
# 7. FINAL EVALUATION
# ============================================================================
print("\n[6/6] Final evaluation...")

# Load best model
model.load_state_dict(torch.load('block1_best.pth'))

train_metrics = validate_model(model, train_loader, device)
val_metrics = validate_model(model, val_loader, device)
test_metrics = validate_model(model, test_loader, device)

print("\n" + "="*80)
print("FINAL SSL TASK PERFORMANCE")
print("="*80)
print(f"Split    | Loss   | Element Acc | Position Acc | Count Acc")
print(f"---------+--------+-------------+--------------+----------")
print(f"TRAIN    | {train_metrics['loss']:.4f} | {train_metrics['element_acc']:.4f}      | {train_metrics['position_acc']:.4f}       | {train_metrics['count_acc']:.4f}")
print(f"VAL      | {val_metrics['loss']:.4f} | {val_metrics['element_acc']:.4f}      | {val_metrics['position_acc']:.4f}       | {val_metrics['count_acc']:.4f}")
print(f"TEST     | {test_metrics['loss']:.4f} | {test_metrics['element_acc']:.4f}      | {test_metrics['position_acc']:.4f}       | {test_metrics['count_acc']:.4f}")

# ============================================================================
# 8. FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("EXTRACTING DENSE FEATURES")
print("="*80)

model.eval()

def extract_features(dataloader):
    """Extract dense features from trained model."""
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            masked = batch['masked'].to(device)
            y_batch = batch['synthesis_label']
            
            feat = model(masked, return_features=True)
            features_list.append(feat.cpu().numpy())
            labels_list.append(y_batch.numpy())
    
    return np.vstack(features_list), np.concatenate(labels_list)

block1_features_train, y_train_check = extract_features(train_loader)
block1_features_val, y_val_check = extract_features(val_loader)
block1_features_test, y_test_check = extract_features(test_loader)

print(f"Extracted features:")
print(f"  Train: {block1_features_train.shape}")
print(f"  Val:   {block1_features_val.shape}")
print(f"  Test:  {block1_features_test.shape}")

# Analyze sparsity reduction
train_sparsity = (block1_features_train == 0).mean() * 100
val_sparsity = (block1_features_val == 0).mean() * 100
print(f"\nSparsity reduction: {97:.1f}% (input) -> {train_sparsity:.1f}% (output)")

# ============================================================================
# 9. DOWNSTREAM TASK EVALUATION (LINEAR PROBE)
# ============================================================================
print("\n" + "="*80)
print("DOWNSTREAM SYNTHESIZABILITY PREDICTION (LINEAR PROBE)")
print("="*80)

# Convert PU labels to binary (1=positive, -1=unknown -> 0)
train_binary = (y_train_check == 1).astype(int)
val_binary = (y_val_check == 1).astype(int)
test_binary = (y_test_check == 1).astype(int)

# Train logistic regression probe
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(block1_features_train, train_binary)

# Evaluate
val_pred = clf.predict_proba(block1_features_val)[:, 1]
test_pred = clf.predict_proba(block1_features_test)[:, 1]

val_auc = roc_auc_score(val_binary, val_pred)
test_auc = roc_auc_score(test_binary, test_pred)

print(f"Linear probe AUC (frozen features):")
print(f"  VAL:  {val_auc:.4f}")
print(f"  TEST: {test_auc:.4f}")

# ============================================================================
# 10. SAVE OUTPUTS
# ============================================================================
print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save extracted features
np.save('block1_features_train.npy', block1_features_train)
np.save('block1_features_val.npy', block1_features_val)
np.save('block1_features_test.npy', block1_features_test)

print("Saved feature files:")
print("  - block1_features_train.npy")
print("  - block1_features_val.npy")
print("  - block1_features_test.npy")
print("  - block1_best.pth (model weights)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("BLOCK 1 TRAINING COMPLETE!")
print("="*80)
print(f"INPUT:  103x4 element matrix (97% sparse)")
print(f"OUTPUT: 64D dense features ({train_sparsity:.1f}% sparse)")
print(f"\nSSL Performance (TEST):")
print(f"  - Position Accuracy:  {test_metrics['position_acc']:.4f} (Target: >= 0.95)")
print(f"  - Count Accuracy:     {test_metrics['count_acc']:.4f} (Target: >= 0.90)")
print(f"\nDownstream Performance:")
print(f"  - Synthesizability AUC: {test_auc:.4f}")
print(f"\nReady for Block 2!")
print("="*80)
