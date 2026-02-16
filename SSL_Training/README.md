# SSL Training - 6-Block Multi-Modal Architecture

This directory contains the training codes for the 6 specialized encoder blocks that comprise the self-supervised learning (SSL) framework.

## Architecture Overview

The SSL framework processes raw FTCP data (129,473 materials, 25,200D) through 6 independent encoder blocks, each specialized for a different aspect of crystallographic structure:

```
Input: FTCP (103, 64) → 25,200D
    ↓
┌─────────────────────────────────────────┐
│  6 Parallel Encoder Blocks (SSL)        │
│                                          │
│  Block 1: Element Matrix      → 256D    │
│  Block 2: Crystal System      → 256D    │
│  Block 3: Atomic Sites        → 512D    │
│  Block 4: Site Occupancy      → 256D    │
│  Block 5: Reciprocal Space    → 512D    │
│  Block 6: Structure Factors   → 256D    │
└─────────────────────────────────────────┘
    ↓
Concatenate + Bottleneck MLP → 2,048D
    ↓
Output: SSL Features (2,048D)
```

## Block Descriptions

### Block 1: Element Matrix Encoder
- **Input**: FTCP[0:103, 0:4] - Element composition (412D)
- **Output**: 256D compressed representation
- **Architecture**: CNN + Attention
- **Pretext Task**: Masked Element Modeling (MEM)
- **Sparsity**: 93.75% (most elements absent in each material)
- **Status**: ❌ **TRAINING CODE NOT FOUND** (will be provided separately)

### Block 2: Crystal System Encoder
- **Input**: FTCP[0:103, 4:8] - Crystal symmetry (412D)
- **Output**: 256D compressed representation
- **Architecture**: Transformer encoder
- **Pretext Task**: Crystal system classification (7 classes)
- **Sparsity**: 99.95% (highly sparse)
- **Status**: ✅ Available (`Block2_Crystal_System/train_block2.py`)

### Block 3: Spatial Graph Network
- **Input**: FTCP[0:103, 8:32] - 3D atomic coordinates (2,472D)
- **Output**: 512D compressed representation
- **Architecture**: Graph Neural Network (GNN)
- **Pretext Tasks**: Coordinate reconstruction, distance prediction
- **Sparsity**: 94.57%
- **Status**: ✅ Available (`Block3_Atomic_Sites/train_block3.py`)

### Block 4: Site Occupancy GAT
- **Input**: FTCP[0:103, 32:40] - Partial occupancy (824D)
- **Output**: 256D compressed representation
- **Architecture**: Graph Attention Network (GAT)
- **Pretext Task**: Occupancy pattern prediction
- **Sparsity**: 98.65%
- **Status**: ✅ Available (`Block4_Site_Occupancy/train_block4.py`)

### Block 5: Reciprocal Space Transformer
- **Input**: FTCP[0:103, 40:56] - k-space features (1,648D)
- **Output**: 512D compressed representation
- **Architecture**: Transformer with reciprocal space attention
- **Pretext Tasks**: Masked k-distance modeling, k-point interpolation
- **Sparsity**: 30.67% (densest block)
- **Status**: ✅ Available (`Block5_Reciprocal_Space/train_block5.py`)

### Block 6: Deep Structure Factor Encoder
- **Input**: FTCP[0:103, 56:63] - Structure factors (721D)
- **Output**: 256D compressed representation
- **Architecture**: CNN + Residual blocks
- **Pretext Tasks**: Structure factor reconstruction, real↔reciprocal mapping
- **Sparsity**: 85.71%
- **Status**: ✅ Available (`Block6_Structure_Factors/train_block6.py`)

## Training Summary

| Block | Input Dim | Output Dim | Pretext Tasks | Status |
|-------|-----------|------------|---------------|--------|
| Block 1 | 412 | 256 | Masked Element Modeling | ❌ Missing |
| Block 2 | 412 | 256 | Crystal System Classification | ✅ Available |
| Block 3 | 2,472 | 512 | Coordinate Reconstruction | ✅ Available |
| Block 4 | 824 | 256 | Occupancy Prediction | ✅ Available |
| Block 5 | 1,648 | 512 | K-distance Modeling | ✅ Available |
| Block 6 | 721 | 256 | Structure Factor Reconstruction | ✅ Available |
| **Total** | **6,489** | **2,048** | **9 tasks total** | **5/6 available** |

## Self-Supervised Pretext Tasks

Each block uses one or more pretext tasks for unsupervised pre-training:

1. **Masked Element Modeling (Block 1)**: Predict masked element identities
2. **Crystal System Classification (Block 2)**: Predict 7 crystal systems
3. **Coordinate Reconstruction (Block 3)**: Reconstruct masked atomic coordinates
4. **Distance Prediction (Block 3)**: Predict pairwise atomic distances
5. **Occupancy Pattern Prediction (Block 4)**: Predict partial occupancy values
6. **Masked K-distance Modeling (Block 5)**: Predict masked reciprocal distances
7. **K-point Interpolation (Block 5)**: Interpolate between k-space points
8. **Structure Factor Reconstruction (Block 6)**: Reconstruct masked structure factors
9. **Real↔Reciprocal Mapping (Block 6)**: Map between real and reciprocal space

## Training Procedure

### Step 1: Pre-train Each Block Independently
Each block trains on its own FTCP slice using self-supervised objectives.

```bash
cd Block2_Crystal_System
python train_block2.py  # Trains Block 2 encoder

cd ../Block3_Atomic_Sites
python train_block3.py  # Trains Block 3 encoder

# ... repeat for blocks 4, 5, 6
```

### Step 2: Extract Features from Pre-trained Encoders
After training, extract 256D or 512D features from each block for all 129,473 materials.

### Step 3: Concatenate and Train Bottleneck MLP
Concatenate all 6 block outputs (total: 2,048D) and optionally train a bottleneck MLP to further compress.

### Step 4: Save Final SSL Features
Export the 2,048D SSL features for downstream tasks (Scenarios 1, 2, 3).

## Training Configuration

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16+ GB VRAM (RTX 3090, A100, V100)
- **RAM**: 32+ GB system memory
- **Storage**: ~30 GB for FTCP data + checkpoints

### Software Requirements
```
torch>=1.10.0
torch-geometric>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

### Hyperparameters (Typical)
- **Batch size**: 256-512
- **Learning rate**: 1e-4 (Adam optimizer)
- **Epochs**: 50-100 (early stopping on validation loss)
- **Dropout**: 0.1-0.2
- **Weight decay**: 1e-5

### Training Time (Approximate)
- Block 1: ~2 hours (missing)
- Block 2: ~1 hour
- Block 3: ~3 hours (largest)
- Block 4: ~1.5 hours
- Block 5: ~2.5 hours
- Block 6: ~1 hour
- **Total**: ~11 hours on single GPU

## Pre-trained Weights

Pre-trained SSL features (2,048D) are available in `../2_Data/SSL_Features/all_ssl_features.npy`.

If you want to skip pre-training and use these features directly, you can proceed to evaluation scenarios (`../3_Evaluation_Scenarios/`).

## Key Design Choices

### Why 6 Separate Blocks?
- Each block specializes in a different crystallographic modality
- Independent training prevents catastrophic interference
- Enables modular architecture (can add/remove blocks)
- Better interpretability (can analyze per-block contributions)

### Why Self-Supervised?
- Labels are expensive (DFT calculations cost ~1 hour per material)
- Unlabeled FTCP data is abundant (millions of materials)
- Pre-training learns general crystallographic patterns
- Transfer to downstream tasks with few labeled samples (Scenario 3)

### Why 2,048D Output?
- Balances compression (12.3× from 25,200D) and information retention
- Enables fast downstream models (linear regression in ~40 seconds)
- Sufficient capacity for diverse property prediction tasks

## Validation

Pre-trained SSL features have been validated on:
1. **Scenario 1**: 8/8 linear regression comparisons (100% success rate)
2. **Scenario 2**: 32.81% SHAP importance (4.36× enrichment)
3. **Scenario 3**: 5.77% contribution at 1% data (few-shot learning)

## Missing Files

❌ **Block 1 training code** (`Block1_Element_Matrix/train_block1.py`)
   - User will provide from another repository
   - Element Matrix Encoder (412D → 256D)
   - Pretext task: Masked Element Modeling

All other blocks (2-6) are available and ready to use.

## Citation

If you use these training codes, please cite:

```bibtex
@article{your_paper_2026,
  title={Self-Supervised Feature Extraction for Crystallographic Materials Property Prediction},
  author={Your Name},
  journal={To be determined},
  year={2026}
}
```

## Next Steps

1. Pre-train the 6 encoder blocks (or use pre-trained features)
2. Proceed to evaluation scenarios (`../3_Evaluation_Scenarios/`)
3. Reproduce the three successful scenarios demonstrating SSL effectiveness
