# Sustainable AI for Medical Diagnostics

**A Multi-Objective Comparative Study of Energy, Emissions, and Accuracy Across Classical, Neuromorphic, and Quantum Architectures**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20|%20snnTorch%20|%20PennyLane-orange)](https://snntorch.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MedMNIST_v2-red)](https://medmnist.com/)

## 📄 Abstract
The escalating environmental footprint of artificial intelligence poses a critical challenge to healthcare sustainability. This work presents a comparative analysis of three computing paradigms for medical image classification:

1.  **Classical CNNs** (Convolutional Neural Networks)
2.  **Neuromorphic SNNs** (Spiking Neural Networks)
3.  **Hybrid Quantum-Classical Models** (QML)

Using the MedMNIST v2 dataset (OrganAMNIST), we introduce frameworks for quantifying the **Accuracy-Carbon-Energy (ACE)** trade-offs, demonstrating that optimized SNNs can achieve competitive accuracy with **46% reduced carbon emissions** compared to classical baselines.

## 💡 Key Contributions

This repository implements the novel multi-objective frameworks introduced in the paper:

* **ACE Index (Accuracy-Carbon-Energy):** A normalized weighted metric for evaluating trade-offs in resource-constrained environments (e.g., remote clinics vs. data centers).
* **Eco-Performance Score (EPS):** A logarithmic metric enabling fair comparison across orders-of-magnitude energy differences.
* **Spike Frequency Modulation:** Analysis code to identify the critical "Green Operating Point" (~4-5% sparsity) where energy savings maximize before accuracy collapse.

## 💻 Hardware Context

To simulate a realistic, heterogeneous deployment scenario, this research was conducted on two distinct hardware tiers:

* **High-Performance Training (Server):** NVIDIA RTX A6000 (48GB VRAM)
* **Edge Inference (Mobile/Clinical):** Intel® Core™ i5-12500H (12 Cores: 4P/8E)

*Note: The code in this repository is hardware-agnostic and will run on standard consumer CPUs/GPUs, but results (specifically runtime and energy) will vary based on your host machine.*

## 📂 Repository Contents

* **`demo_pipeline.ipynb`**: A self-contained Jupyter Notebook that defines the model architectures, data loaders, and a "toy" training loop to verify pipeline functionality.
* **`requirements.txt`**: List of dependencies required to run the environment.

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook or Google Colab

### Installation

```bash
git clone [https://github.com/your-username/Sustainable-AI-Medical-Diagnostics.git](https://github.com/your-username/Sustainable-AI-Medical-Diagnostics.git)
cd Sustainable-AI-Medical-Diagnostics
pip install -r requirements.txt