
## Enriching FTCP Crystal Representations via Multi-Modal Self-Supervised Learning

---

## 📘 Overview

This repository provides the codebase, evaluation scripts, and experimental structure for the paper:

> **Enriching FTCP Representation of Crystals through Multi-Modal Self-Supervised Learning for Enhanced Materials Property Prediction**

The work introduces a hierarchical multi-modal self-supervised learning (SSL) framework that enriches the high-dimensional Fourier-Transformed Crystallographic Properties (FTCP) representation into compact, information-dense features without using any property labels.

The framework decomposes FTCP into six physically motivated modalities, trains specialized neural architectures via 13 self-supervised pretext tasks, and produces enriched representations that generalize across downstream materials property prediction tasks.

---

## 🔬 Key Contributions

- Multi-modal SSL framework tailored to crystallographic data
- **12.3× dimensionality reduction** (25,200 → 2,048) without information loss
- Six specialized neural architectures aligned with physical structure
- 13 physics-informed pretext tasks (masking, reconstruction, consistency)
- Robust downstream evaluation across three complementary scenarios:
  - Linear regression stability
  - Hybrid FTCP + SSL interpretability (SHAP)
  - Extreme low-data transfer learning (1% labeled data)

---

## 📂 Repository Structure

```
Crys_Rep_Enrich/
│
├── Data/                       # (Empty placeholders – see Dataset section)
│   ├── FTCP/
│   ├── Labels/
│   └── Splitted_Data/
│       ├── Split_10Test_90Train/
│       ├── Split_20Test_80Train/
│       ├── Split_30Test_70Train/
│       └── Split_40Test_60Train/
│
├── SSL_Training/               # Self-supervised pretraining (six FTCP blocks)
│   ├── Block1_Element_Matrix/
│   ├── Block2_Crystal_System/
│   ├── Block3_Atomic_Sites/
│   ├── Block4_Site_Occupancy/
│   ├── Block5_Reciprocal_Space/
│   └── Block6_Structure_Factors/
│
├── Evaluation_Scenarios/       # Downstream evaluation framework
│   ├── Scenario1_Linear_Regression/
│   ├── Scenario2_Hybrid_Enrichment/
│   └── Scenario3_Sample_Efficiency/
│
├── Supplemental Information/   # Figures and additional results
│
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 📦 Dataset Access

> ⚠️ The `Data/` folders in this repository are intentionally empty.

Due to the large size of the dataset (FTCP tensors, splits, and labels), all data is hosted externally.

**🔗 Download Dataset:**
👉 [Hugging Face Dataset Repository](https://huggingface.co/datasets/danial199472/Crys_Rep_Enrich)

After downloading and extracting, place the contents into the corresponding `Data/` subfolders without modifying the directory names.

---

## 🧠 FTCP Modalities and SSL Blocks

| Block | FTCP Modality | Physical Meaning |
|-------|--------------|-----------------|
| 1 | Element Composition Matrix | Chemical identity & sparsity |
| 2 | Lattice Parameters | Symmetry & geometry |
| 3 | Atomic Sites | Local coordination & packing |
| 4 | Site Occupancy | Disorder & partial occupancy |
| 5 | Reciprocal Space | k-point topology |
| 6 | Structure Factors | Diffraction & Fourier physics |

Each block is trained independently using tailored architectures and physics-informed self-supervised objectives.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

### GPU Support

For CUDA-enabled PyTorch, follow instructions at: https://pytorch.org/

Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 💻 System Recommendations

- **Python:** 3.8+
- **RAM:** ≥ 32 GB (recommended)
- **GPU:** Optional, recommended for SSL training
- **Storage:** ≥ 150 GB for full dataset and outputs

---

## 📬 Contact

**Corresponding Author:**
Yaser Mike Banad — bana@ou.edu

**First Author:**
Danial Ebrahimzadeh — danial.ebrahimzadeh@ou.edu

