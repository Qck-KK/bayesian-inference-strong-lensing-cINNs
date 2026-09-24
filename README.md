# Bayesian Inference of Strong Lensing with cINNs

**Course:** Generative Neural Networks for the Sciences (Winter Semester 2025/26)  
**Institution:** Heidelberg University  
**Authors:** Zhiheng Lin ([@Zhiheng-SC](https://github.com/Zhiheng-SC)), Zhikai Zhang ([@Qck-KK](https://github.com/Qck-KK)), Yimin Yan ([@Yimin-Yan](https://github.com/Yimin-Yan))  

---

## 📖 Project Overview
This repository contains the official PyTorch implementation for our final project on **Simulation-Based Inference (SBI) for Strong Gravitational Lensing**. 

Inferring continuous lens mass parameters from noisy telescope observations is a fundamentally ill-posed inverse problem plagued by complex parameter degeneracies (e.g., the mass-sheet degeneracy). To resolve this, we implemented a **Conditional Invertible Neural Network (cINN)** based on the RealNVP architecture, conditioned on spatial features extracted by a custom 4-layer Convolutional Neural Network (CNN). 

Our pipeline bypasses the computational bottlenecks of traditional MCMC and optimization algorithms, delivering mathematically exact full Bayesian posterior distributions $p(\theta|x_{obs})$ in fractions of a second, while autonomously detecting Out-of-Distribution (OOD) astronomical anomalies.

### ✨ Key Scientific Highlights
- **Amortized Inference:** Achieves a ~26x absolute speedup (and orders of magnitude effective speedup for posterior sampling) compared to traditional Nelder-Mead optimization.
- **Exact Posterior Calibration:** Statistically validated using empirical coverage curves and rigorous Simulation-Based Calibration (SBC) rank histograms.
- **Robust Anomaly Detection:** Leverages exact latent space $\chi^2_3$ statistics to achieve a perfect AUC of 1.000 against extreme noise, PSF degradation, and high-mass extrapolations.

---

## ⚙️ Dependencies & Installation
The codebase is built with Python 3.9+ and PyTorch. The physical forward simulator relies on `lenstronomy`. 

To set up the environment, run:
```bash
pip install torch torchvision numpy scipy h5py matplotlib corner scikit-learn lenstronomy scikit-image
```

Or equivalently:
```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure
| File | Description |
|---|---|
| `Strong_Lensing_Simulator.ipynb` | Full pipeline: simulator, data generation, cINN training, and evaluation |
| `FinalReport.pdf` | Final project report |
| `requirements.txt` | Python dependencies |

---

## 🚀 Usage
All steps are contained in `Strong_Lensing_Simulator.ipynb` and should be run top to bottom.

1. **Generate the training data.** The HDF5 datasets are not included in this repository (the training set is ~780 MB). Running the *"Mass Production and Data Storage (HDF5)"* section creates `lensing_full_train_50k.h5` (50,000 simulated 64×64 lensing images; takes about a minute).
2. **Generate the OOD test sets.** The *"Handling the Sim-to-Real Gap: OOD Testing"* section creates `ood_extreme_noise.h5`, `ood_psf_blur.h5`, and `ood_high_mass.h5`.
3. **Train the model.** The *"Joint training"* section trains the summary CNN and the cINN jointly and saves `lensing_sbi_joint_model.pth` (about 10 minutes on a GPU). The weights are not included in this repository.
4. **Evaluate.** The *"Model Validation and Experimental Analysis"* section automatically generates the independent test set `lensing_independent_test_1k.h5` if it is missing, then runs the calibration, posterior predictive checks, OOD detection (ROC), SBC, and the Nelder-Mead baseline comparison.

A CUDA-capable GPU is recommended for training; evaluation also runs on CPU.

---

## 📄 License
This project is released under the [MIT License](LICENSE).
