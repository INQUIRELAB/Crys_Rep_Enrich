#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🌊 BLOCK 5: RECIPROCAL SPACE PROCESSING (CORRECTED)")
print("K-Point Distances → Self-Supervised Learning → Dense Features")
print("Row 305 with actual k-point distance data")
print("=" * 80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

class ReciprocalSpaceDataset(Dataset):
    def __init__(self, k_distances, y_synth=None, augment=True):
        """
        Dataset for Reciprocal Space Self-Supervised Learning
        
        Args:
            k_distances: (N, 59) k-point distances from origin in reciprocal space
            y_synth: Synthesis labels (for final evaluation only)
            augment: Whether to apply data augmentation
        """
        self.k_distances = torch.FloatTensor(k_distances)
        self.y_synth = torch.LongTensor(y_synth) if y_synth is not None else None
        self.augment = augment
        
        print(f"🔬 Dataset created: {len(self.k_distances)} samples")
        
        # Analyze k-point patterns
        self._analyze_k_patterns()
        
    def _analyze_k_patterns(self):
        """Analyze k-point distance patterns for better understanding"""
        k_flat = self.k_distances.reshape(-1)
        
        # Find non-zero k-points
        non_zero_mask = k_flat > 1e-6
        active_k = k_flat[non_zero_mask]
        
        self.k_stats = {
            'total_k_points': len(k_flat),
            'active_k_points': len(active_k),
            'sparsity_pct': (1 - len(active_k) / len(k_flat)) * 100,
            'k_range': [active_k.min().item(), active_k.max().item()] if len(active_k) > 0 else [0, 0],
            'k_mean': active_k.mean().item() if len(active_k) > 0 else 0,
            'k_std': active_k.std().item() if len(active_k) > 0 else 0,
            'unique_k_values': len(torch.unique(active_k)) if len(active_k) > 0 else 0
        }
        
        print(f"📊 K-Point Analysis:")
        print(f"   Total k-points per material: {self.k_distances.shape[1]}")
        print(f"   Active k-points: {self.k_stats['active_k_points']:,} / {self.k_stats['total_k_points']:,}")
        print(f"   Sparsity: {self.k_stats['sparsity_pct']:.2f}%")
        print(f"   K-distance range: [{self.k_stats['k_range'][0]:.4f}, {self.k_stats['k_range'][1]:.4f}] Å⁻¹")
        print(f"   Mean k-distance: {self.k_stats['k_mean']:.4f} ± {self.k_stats['k_std']:.4f} Å⁻¹")
        print(f"   Unique k-values: {self.k_stats['unique_k_values']}")
        
    def __len__(self):
        return len(self.k_distances)
    
    def __getitem__(self, idx):
        k_dists = self.k_distances[idx]  # (59,)
        
        # Apply augmentation during training
        if self.augment:
            k_dists = self._augment_k_distances(k_dists)
            
        # Create self-supervised tasks
        mkm_data = self._create_masked_k_modeling_task(k_dists)
        kos_data = self._create_k_ordering_task(k_dists)
        sfr_data = self._create_structure_factor_task(k_dists)
        
        return {
            'k_distances': k_dists,
            'mkm': mkm_data,
            'kos': kos_data, 
            'sfr': sfr_data,
            'synth_label': self.y_synth[idx] if self.y_synth is not None else -1
        }
    
    def _augment_k_distances(self, k_dists):
        """Apply k-point distance augmentation"""
        # Small multiplicative noise for non-zero values only
        if torch.rand(1) < 0.3:
            noise_factor = 1.0 + torch.randn(1) * 0.02  # ±2% multiplicative noise
            non_zero_mask = k_dists > 1e-6
            k_dists[non_zero_mask] = k_dists[non_zero_mask] * noise_factor
            
        # Random permutation of equivalent k-points (same magnitude)
        if torch.rand(1) < 0.2:
            non_zero_k = k_dists[k_dists > 1e-6]
            if len(non_zero_k) > 1:
                unique_k, inverse_indices = torch.unique(non_zero_k, return_inverse=True)
                for uk in unique_k:
                    mask = (k_dists == uk)
                    if mask.sum() > 1:  # Multiple k-points with same magnitude
                        indices = torch.where(mask)[0]
                        shuffled_indices = indices[torch.randperm(len(indices))]
                        k_dists[mask] = k_dists[shuffled_indices]
                    
        return k_dists
    
    def _create_masked_k_modeling_task(self, k_dists):
        """Create Masked K-point Modeling (MKM) task"""
        # Only mask non-zero k-points
        non_zero_mask = k_dists > 1e-6
        non_zero_indices = torch.where(non_zero_mask)[0]
        
        if len(non_zero_indices) == 0:
            # If no non-zero k-points, create dummy task
            return {
                'masked_k': k_dists.clone(),
                'mask_indices': torch.tensor([], dtype=torch.long),
                'target_values': torch.tensor([], dtype=torch.float)
            }
        
        # Mask 15% of non-zero k-points
        n_mask = max(1, int(0.15 * len(non_zero_indices)))
        selected_indices = non_zero_indices[torch.randperm(len(non_zero_indices))[:n_mask]]
        
        # Create masked k-distances
        masked_k = k_dists.clone()
        original_values = k_dists[selected_indices].clone()
        masked_k[selected_indices] = 0.0  # Mask with zeros
        
        return {
            'masked_k': masked_k,
            'mask_indices': selected_indices,
            'target_values': original_values
        }
    
    def _create_k_ordering_task(self, k_dists):
        """Create K-point Ordering/Sequence (KOS) task"""
        # Only consider non-zero k-points for ordering
        non_zero_mask = k_dists > 1e-6
        non_zero_k = k_dists[non_zero_mask]
        
        if len(non_zero_k) == 0:
            # If no non-zero k-points, assign all to class 0
            ordering_labels = torch.zeros(len(k_dists), dtype=torch.long)
        else:
            # Create ordering classes based on k-magnitude ranges
            k_min, k_max = non_zero_k.min(), non_zero_k.max()
            k_range = k_max - k_min
            
            ordering_labels = torch.zeros(len(k_dists), dtype=torch.long)
            
            # Only classify non-zero k-points
            for i, k in enumerate(k_dists):
                if k > 1e-6:  # Non-zero k-point
                    if k_range > 1e-6:  # Valid range
                        if k <= k_min + 0.25 * k_range:
                            ordering_labels[i] = 0  # Very small k
                        elif k <= k_min + 0.50 * k_range:
                            ordering_labels[i] = 1  # Small k
                        elif k <= k_min + 0.75 * k_range:
                            ordering_labels[i] = 2  # Medium k  
                        else:
                            ordering_labels[i] = 3  # Large k
                    else:
                        ordering_labels[i] = 1  # Default to small if no range
                # else: keep as 0 for zero k-points
                
        return {
            'ordering_labels': ordering_labels
        }
    
    def _create_structure_factor_task(self, k_dists):
        """Create Structure Factor Reconstruction (SFR) task"""
        # Simulate structure factors based on k-distances
        structure_factors = torch.zeros_like(k_dists)
        
        # Only compute for non-zero k-points
        non_zero_mask = k_dists > 1e-6
        non_zero_k = k_dists[non_zero_mask]
        
        if len(non_zero_k) > 0:
            # Structure factor magnitude typically decreases with |k|
            # F(k) ∝ exp(-B|k|²/4) for Debye-Waller factor
            B_factor = 0.1  # Thermal parameter (smaller to preserve signal)
            sf_values = torch.exp(-B_factor * non_zero_k**2 / 4)
            
            # Add some crystallographic modulation
            for i, k in enumerate(non_zero_k):
                # Simulate allowed/forbidden reflections
                if torch.rand(1) < 0.05:  # 5% systematic absences
                    sf_values[i] *= 0.1
                # Add some texture effects
                sf_values[i] *= (1 + 0.2 * torch.sin(k * 5))
                
            # Normalize and assign to structure_factors
            if sf_values.max() > 0:
                sf_values = sf_values / sf_values.max()
                structure_factors[non_zero_mask] = sf_values
        
        return {
            'structure_factors': structure_factors
        }

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, max_freq=10.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # Create frequency basis
        freqs = torch.linspace(1.0, max_freq, embed_dim // 4)
        self.register_buffer('freqs', freqs)
        
        # Linear projection
        self.linear = nn.Linear(input_dim, embed_dim // 4)
        
    def forward(self, x):
        # x: (batch_size, seq_len) or (batch_size, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
        # Linear features
        linear_features = self.linear(x)  # (batch_size, seq_len, embed_dim//4)
        
        # Fourier features (only for non-zero values)
        x_scaled = x * self.freqs.unsqueeze(0).unsqueeze(0)  # (batch_size, seq_len, embed_dim//4)
        sin_features = torch.sin(2 * np.pi * x_scaled)
        cos_features = torch.cos(2 * np.pi * x_scaled)
        
        # Concatenate all features
        fourier_features = torch.cat([
            linear_features,
            sin_features,
            cos_features,
            x.repeat(1, 1, self.embed_dim // 4)  # Raw features
        ], dim=-1)
        
        return fourier_features

class ReciprocalTransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=256):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights

class ReciprocalSpaceNet(nn.Module):
    def __init__(self, seq_len=59, embed_dim=64, num_layers=2, num_heads=4, output_dim=256):
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Fourier embedding for k-distances
        self.fourier_embed = FourierEmbedding(input_dim=1, embed_dim=embed_dim)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.1)
        
        # Mini-transformer layers
        self.transformer_blocks = nn.ModuleList([
            ReciprocalTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Task-specific heads
        self.mkm_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Predict single k-distance
        )
        
        self.kos_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4)  # 4 ordering classes
        )
        
        self.sfr_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Predict structure factor
        )
        
        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, k_distances, return_features=False):
        batch_size = k_distances.shape[0]
        
        # Fourier embedding
        x = self.fourier_embed(k_distances)  # (batch_size, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Create attention mask for zero k-points
        mask = (k_distances <= 1e-6)  # True for positions to ignore
        
        # Transformer processing
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask=mask)
            attention_weights.append(attn_weights)
            
        if return_features:
            # Global pooling for final features (ignoring masked positions)
            non_zero_mask = ~mask  # (batch_size, seq_len)
            
            # Weighted average pooling (only non-zero positions)
            pooled_features = []
            for i in range(batch_size):
                valid_positions = non_zero_mask[i]
                if valid_positions.sum() > 0:
                    pooled = x[i][valid_positions].mean(dim=0)
                else:
                    pooled = x[i].mean(dim=0)  # Fallback
                pooled_features.append(pooled)
            
            pooled = torch.stack(pooled_features)  # (batch_size, embed_dim)
            return self.feature_proj(pooled)
        
        # Task predictions
        mkm_pred = self.mkm_head(x)  # (batch_size, seq_len, 1)
        kos_pred = self.kos_head(x)  # (batch_size, seq_len, 4)
        sfr_pred = self.sfr_head(x)  # (batch_size, seq_len, 1)
        
        # Global pooling for final features (ignoring masked positions)
        non_zero_mask = ~mask  # (batch_size, seq_len)
        pooled_features = []
        for i in range(batch_size):
            valid_positions = non_zero_mask[i]
            if valid_positions.sum() > 0:
                pooled = x[i][valid_positions].mean(dim=0)
            else:
                pooled = x[i].mean(dim=0)  # Fallback
            pooled_features.append(pooled)
        
        pooled = torch.stack(pooled_features)  # (batch_size, embed_dim)
        features = self.feature_proj(pooled)
        
        return {
            'mkm_pred': mkm_pred.squeeze(-1),  # (batch_size, seq_len)
            'kos_pred': kos_pred,              # (batch_size, seq_len, 4)
            'sfr_pred': sfr_pred.squeeze(-1),  # (batch_size, seq_len)
            'features': features,              # (batch_size, output_dim)
            'attention_weights': attention_weights
        }

def custom_collate(batch):
    """Custom collate function for batch processing"""
    k_distances = torch.stack([item['k_distances'] for item in batch])
    
    # Collate MKM data
    mkm_masked_k = torch.stack([item['mkm']['masked_k'] for item in batch])
    mkm_mask_indices = [item['mkm']['mask_indices'] for item in batch]
    mkm_target_values = [item['mkm']['target_values'] for item in batch]
    
    # Collate KOS data
    kos_labels = torch.stack([item['kos']['ordering_labels'] for item in batch])
    
    # Collate SFR data
    sfr_targets = torch.stack([item['sfr']['structure_factors'] for item in batch])
    
    # Synthesis labels
    synth_labels = torch.tensor([item['synth_label'] for item in batch])
    
    return {
        'k_distances': k_distances,
        'mkm': {
            'masked_k': mkm_masked_k,
            'mask_indices': mkm_mask_indices,
            'target_values': mkm_target_values
        },
        'kos': {'labels': kos_labels},
        'sfr': {'targets': sfr_targets},
        'synth_labels': synth_labels
    }

def compute_mkm_loss(mkm_pred, mkm_data):
    """Compute Masked K-point Modeling loss"""
    total_loss = 0.0
    count = 0
    
    for i, (mask_indices, target_values) in enumerate(zip(mkm_data['mask_indices'], mkm_data['target_values'])):
        if len(mask_indices) > 0:
            pred_values = mkm_pred[i][mask_indices]
            target_values = target_values.to(pred_values.device)
            loss = F.mse_loss(pred_values, target_values)
            total_loss += loss
            count += 1
            
    return total_loss / max(count, 1)

def compute_kos_loss(kos_pred, kos_data):
    """Compute K-point Ordering/Sequence loss"""
    kos_labels = kos_data['labels'].to(kos_pred.device)
    
    # Reshape for cross-entropy
    kos_pred_flat = kos_pred.view(-1, 4)  # (batch_size * seq_len, 4)
    kos_labels_flat = kos_labels.view(-1)  # (batch_size * seq_len)
    
    return F.cross_entropy(kos_pred_flat, kos_labels_flat)

def compute_sfr_loss(sfr_pred, sfr_data):
    """Compute Structure Factor Reconstruction loss"""
    sfr_targets = sfr_data['targets'].to(sfr_pred.device)
    return F.mse_loss(sfr_pred, sfr_targets)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    mkm_losses, kos_losses, sfr_losses = [], [], []
    kos_correct = 0
    kos_total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        k_distances = batch['k_distances'].to(device)
        
        # Forward pass
        outputs = model(k_distances)
        
        # Compute individual losses
        mkm_loss = compute_mkm_loss(outputs['mkm_pred'], batch['mkm'])
        kos_loss = compute_kos_loss(outputs['kos_pred'], batch['kos'])
        sfr_loss = compute_sfr_loss(outputs['sfr_pred'], batch['sfr'])
        
        # Combined loss with weighting
        combined_loss = 0.4 * mkm_loss + 0.3 * kos_loss + 0.3 * sfr_loss
        
        # Backward pass
        optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += combined_loss.item()
        mkm_losses.append(mkm_loss.item())
        kos_losses.append(kos_loss.item())
        sfr_losses.append(sfr_loss.item())
        
        # KOS accuracy
        kos_pred_classes = outputs['kos_pred'].argmax(dim=-1)
        kos_correct += (kos_pred_classes == batch['kos']['labels'].to(device)).float().sum().item()
        kos_total += kos_pred_classes.numel()
        
        pbar.set_postfix({
            'Loss': f'{combined_loss.item():.4f}',
            'MKM': f'{mkm_loss.item():.4f}',
            'KOS': f'{kos_loss.item():.4f}',
            'SFR': f'{sfr_loss.item():.4f}'
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mkm_loss': np.mean(mkm_losses),
        'kos_loss': np.mean(kos_losses),
        'sfr_loss': np.mean(sfr_losses),
        'kos_accuracy': kos_correct / kos_total
    }

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    mkm_losses, kos_losses, sfr_losses = [], [], []
    kos_correct = 0
    kos_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            k_distances = batch['k_distances'].to(device)
            
            # Forward pass
            outputs = model(k_distances)
            
            # Compute individual losses
            mkm_loss = compute_mkm_loss(outputs['mkm_pred'], batch['mkm'])
            kos_loss = compute_kos_loss(outputs['kos_pred'], batch['kos'])
            sfr_loss = compute_sfr_loss(outputs['sfr_pred'], batch['sfr'])
            
            # Combined loss
            combined_loss = 0.4 * mkm_loss + 0.3 * kos_loss + 0.3 * sfr_loss
            
            # Accumulate losses
            total_loss += combined_loss.item()
            mkm_losses.append(mkm_loss.item())
            kos_losses.append(kos_loss.item())
            sfr_losses.append(sfr_loss.item())
            
            # KOS accuracy
            kos_pred_classes = outputs['kos_pred'].argmax(dim=-1)
            kos_correct += (kos_pred_classes == batch['kos']['labels'].to(device)).float().sum().item()
            kos_total += kos_pred_classes.numel()
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mkm_loss': np.mean(mkm_losses),
        'kos_loss': np.mean(kos_losses),
        'sfr_loss': np.mean(sfr_losses),
        'kos_accuracy': kos_correct / kos_total
    }

# Load datasets
print("\n📂 Loading datasets...")
X_train = np.load('X_train_final.npy')
X_val = np.load('X_validation_final.npy')
X_test = np.load('X_test_final.npy')
y_train = np.load('y_train_final.npy')
y_val = np.load('y_validation_final.npy')
y_test = np.load('y_test_final.npy')

print(f"✅ Datasets loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

# Extract k-point distances: Row 305 (CORRECTED - using index 305, not 304!)
print("\n🌊 Extracting k-point distances (Row 305, index 305)...")
k_train = X_train[:, 305, :59]  # CORRECTED: Use index 305, first 59 values
k_val = X_val[:, 305, :59]
k_test = X_test[:, 305, :59]

print(f"K-distance shapes: Train {k_train.shape}, Val {k_val.shape}, Test {k_test.shape}")

# Create datasets
train_dataset = ReciprocalSpaceDataset(k_train, y_train, augment=True)
val_dataset = ReciprocalSpaceDataset(k_val, y_val, augment=False)
test_dataset = ReciprocalSpaceDataset(k_test, y_test, augment=False)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         collate_fn=custom_collate, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       collate_fn=custom_collate, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=custom_collate, num_workers=2)

print(f"✅ Dataloaders created - Batch size: {batch_size}")

# Initialize model
model = ReciprocalSpaceNet(seq_len=59, embed_dim=64, num_layers=2, num_heads=4, output_dim=256).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15)

print(f"\n🏋️  Model Configuration:")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Embed dim: 64, Layers: 2, Heads: 4, Output: 256D")
print(f"   Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")

# Training targets
print(f"\n🎯 Self-Supervised Learning Targets:")
print(f"   MKM (Masked K-point Modeling): MSE < 0.001")
print(f"   KOS (K-point Ordering/Sequence): Accuracy > 70%")
print(f"   SFR (Structure Factor Reconstruction): MSE < 0.05")
print(f"   Synthesis prediction (individual): AUC > 0.60")

# Training loop
num_epochs = 15
best_val_loss = float('inf')
train_history = []
val_history = []

print(f"\n🚀 Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train
    train_metrics = train_epoch(model, train_loader, optimizer, device)
    
    # Validate
    val_metrics = validate_epoch(model, val_loader, device)
    
    # Update scheduler
    scheduler.step()
    
    # Store metrics
    train_history.append(train_metrics)
    val_history.append(val_metrics)
    
    # Print metrics
    print(f"Train - Loss: {train_metrics['total_loss']:.4f}, MKM: {train_metrics['mkm_loss']:.4f}, "
          f"KOS: {train_metrics['kos_loss']:.4f}, SFR: {train_metrics['sfr_loss']:.4f}, "
          f"KOS Acc: {train_metrics['kos_accuracy']:.3f}")
    print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, MKM: {val_metrics['mkm_loss']:.4f}, "
          f"KOS: {val_metrics['kos_loss']:.4f}, SFR: {val_metrics['sfr_loss']:.4f}, "
          f"KOS Acc: {val_metrics['kos_accuracy']:.3f}")
    
    # Save best model
    if val_metrics['total_loss'] < best_val_loss:
        best_val_loss = val_metrics['total_loss']
        torch.save(model.state_dict(), 'Step2_Block5_reciprocal_space_best.pth')
        print(f"✅ New best model saved (val_loss: {best_val_loss:.4f})")

print(f"\n🎉 Training completed!")

# Load best model for evaluation
model.load_state_dict(torch.load('Step2_Block5_reciprocal_space_best.pth'))

# Extract features
print(f"\n🔍 Extracting features from best model...")

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            k_distances = batch['k_distances'].to(device)
            features = model(k_distances, return_features=True)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(batch['synth_labels'].numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)

h5_train, y_train_used = extract_features(model, train_loader, device)
h5_val, y_val_used = extract_features(model, val_loader, device)
h5_test, y_test_used = extract_features(model, test_loader, device)

print(f"✅ Features extracted:")
print(f"   Train: {h5_train.shape}")
print(f"   Val: {h5_val.shape}")
print(f"   Test: {h5_test.shape}")

# Save features
np.save('Step2_Block5_features_train.npy', h5_train)
np.save('Step2_Block5_features_val.npy', h5_val)
np.save('Step2_Block5_features_test.npy', h5_test)
print(f"💾 Features saved to .npy files")

# Evaluate synthesis prediction
print(f"\n📊 Evaluating synthesis prediction performance...")

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(h5_train, y_train_used)

y_train_pred = clf.predict_proba(h5_train)[:, 1]
y_val_pred = clf.predict_proba(h5_val)[:, 1]
y_test_pred = clf.predict_proba(h5_test)[:, 1]

auc_train = roc_auc_score(y_train_used, y_train_pred)
auc_val = roc_auc_score(y_val_used, y_val_pred)
auc_test = roc_auc_score(y_test_used, y_test_pred)

print(f"\n🎯 SYNTHESIS PREDICTION RESULTS:")
print(f"   Train AUC: {auc_train:.4f}")
print(f"   Val AUC: {auc_val:.4f}")
print(f"   Test AUC: {auc_test:.4f}")
print(f"   Target: >0.60 | Gap: {auc_test - 0.60:.3f}")

# Evaluate self-supervised tasks
print(f"\n📈 SELF-SUPERVISED LEARNING EVALUATION:")

final_train = train_history[-1]
final_val = val_history[-1]

print(f"\n✅ FINAL PERFORMANCE:")
print(f"   MKM Loss: {final_val['mkm_loss']:.6f} (Target: <0.001)")
print(f"   KOS Accuracy: {final_val['kos_accuracy']:.3f} (Target: >0.70)")
print(f"   SFR Loss: {final_val['sfr_loss']:.6f} (Target: <0.05)")

mkm_status = "✅" if final_val['mkm_loss'] < 0.001 else "⚠️"
kos_status = "✅" if final_val['kos_accuracy'] > 0.70 else "⚠️"
sfr_status = "✅" if final_val['sfr_loss'] < 0.05 else "⚠️"
auc_status = "✅" if auc_test > 0.60 else "⚠️"

print(f"\n🎯 TARGET ACHIEVEMENT:")
print(f"   {mkm_status} MKM Task: {'ACHIEVED' if final_val['mkm_loss'] < 0.001 else 'NEEDS IMPROVEMENT'}")
print(f"   {kos_status} KOS Task: {'ACHIEVED' if final_val['kos_accuracy'] > 0.70 else 'NEEDS IMPROVEMENT'}")
print(f"   {sfr_status} SFR Task: {'ACHIEVED' if final_val['sfr_loss'] < 0.05 else 'NEEDS IMPROVEMENT'}")
print(f"   {auc_status} Synthesis AUC: {'ACHIEVED' if auc_test > 0.60 else 'NEEDS IMPROVEMENT'}")

print(f"\n" + "=" * 80)
print(f"✅ BLOCK 5 RECIPROCAL SPACE PROCESSING COMPLETED!")
print(f"=" * 80)

print(f"""
🎯 SUMMARY:

SELF-SUPERVISED LEARNING PERFORMANCE:
├── MKM (Masked K-point Modeling): {final_val['mkm_loss']:.6f} MSE {mkm_status}
├── KOS (K-point Ordering): {final_val['kos_accuracy']:.3f} accuracy {kos_status}
├── SFR (Structure Factor Reconstruction): {final_val['sfr_loss']:.6f} MSE {sfr_status}
└── Combined: 3 complementary tasks learning reciprocal space physics

SYNTHESIS PREDICTION:
├── Train AUC: {auc_train:.4f}
├── Val AUC: {auc_val:.4f}
├── Test AUC: {auc_test:.4f} {auc_status}
└── Individual performance: {'Above' if auc_test > 0.60 else 'Below'} target

FEATURES READY:
├── h₅ ∈ ℝ²⁵⁶ reciprocal space representations
├── Samples: {len(h5_train):,} train, {len(h5_val):,} val, {len(h5_test):,} test
└── Multi-modal fusion ready: h₁ + h₂ + h₃ + h₄ + h₅ → 1280D

NEXT STEP: Multi-block feature fusion for target AUC >0.73! 🚀
""") 