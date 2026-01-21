# 🧠 NPMFF‑Net: A Training‑Free Unified Framework for Point Cloud Classification and Segmentation

This repository contains the open‑source implementation of **NPMFF‑Net**, a **training‑free unified framework** for 3D point cloud classification and segmentation, based on the article published in *Knowledge‑Based Systems*. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015680))

---

## 📘 Paper Overview

Point cloud understanding (classification & segmentation) is crucial in areas such as robotics, autonomous navigation, and industrial automation. However, many existing methods rely on **heavy training** and **learnable parameters**, which leads to high computational cost and limited deployment adaptability.

To address this, the paper proposes **NPMFF‑Net**, a **non‑parametric, training‑free architecture** that leverages geometric and frequency information to achieve competitive performance **without model training**. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015680))

---

## 💡 Key Features

- ⭐ **Training‑free:** No learnable parameters or gradient optimization required.  
- 🚀 **Unified Framework:** Handles both **classification** and **segmentation** of point clouds.  
- 📊 **Efficient & Fast:** Suitable for real‑time and low‑resource environments.  
- 🧩 **Effective Geometric Encoding:** Uses *Plücker coordinates* and *Fourier Feature Mapping*.  
- 🤝 **Modular Design:** Easy to integrate into existing systems. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015680))

---

## 🛠️ Repository Contents

| Folder/File | Description |
|-------------|-------------|
| `src/` | Core implementation of NPMFF‑Net |
| `datasets/` | Scripts for handling benchmark datasets |
| `examples/` | Example usage for classification & segmentation |
| `results/` | Evaluation results and visualization |
| `requirements.txt` | Python dependency list |
| `LICENSE` | Open‑source license |

---

## 📦 Installation

Recommended Python version: **3.8+**

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
pip install -r requirements.txt
