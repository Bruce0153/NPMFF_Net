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

## 🚀 Quick Start


### 🔹 Classification Example


### 🔹 Segmentation Example

---

## 📈 Results

| Dataset | Task | Metric | Performance |
|---------|------|--------|-------------|
| ModelNet40 | Classification | Overall Accuracy | *e.g., 90.0%* |
| ShapeNetPart | Segmentation | mIoU | *e.g., 73.1%* |

> These results are achieved **without any training**, demonstrating the potential of training‑free methods. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015680))

---

## 📚 Citation

If you use this code in your work, please cite:

```
@article{zeng2025npmffnet,
  title={NPMFF‑Net: A training‑free unified framework for point cloud classification and segmentation},
  author={Zeng, Hualong and Zhu, Haijiang and Yu, Huaiyuan and Liu, Mengting and An, Ning},
  journal={Knowledge‑Based Systems},
  volume={330},
  pages={114529},
  year={2025},
  publisher={Elsevier}
}
```

---

## 💬 Contact

Developed by Hualong Zeng  
📧 Email: 2024200768@buct.edu.cn
🌐 GitHub: https://github.com/Bruce0153
