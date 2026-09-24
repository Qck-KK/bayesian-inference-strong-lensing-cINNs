# Bayesian Inference of Strong Lensing with cINNs

**Course:** Generative Neural Networks for the Sciences (Winter Semester 2025/26)  
**Institution:** Heidelberg University  
**Authors:** Zhiheng Lin ([@Zhiheng-SC](https://github.com/Zhiheng-SC)), Zhikai Zhang ([@Qck-KK](https://github.com/Qck-KK)), Yimin Yan ([@Yimin-Yan](https://github.com/Yimin-Yan))  
**Post-submission revision (v1.1):** Zhikai Zhang ([@Qck-KK](https://github.com/Qck-KK)): diagnosed and fixed the posterior calibration and OOD evaluation issues (see [Fix History](#-fix-history))  

---

## 📖 Project Overview
This repository contains the official PyTorch implementation for our final project on **Simulation-Based Inference (SBI) for Strong Gravitational Lensing**. 

Inferring continuous lens mass parameters from noisy telescope observations is a fundamentally ill-posed inverse problem plagued by complex parameter degeneracies (e.g., the mass-sheet degeneracy). To resolve this, we implemented a **Conditional Invertible Neural Network (cINN)** based on the RealNVP architecture, conditioned on spatial features extracted by a custom 4-layer Convolutional Neural Network (CNN). 

Once trained, the network returns samples from an approximate posterior $p(\theta|x_{obs})$ in fractions of a second per observation, without re-running an optimizer or MCMC chain for each new image. We also study whether the latent space can flag Out-of-Distribution (OOD) observations.

### ✨ Key Points
- **Calibrated Posteriors:** On 1,000 independent test simulations, the empirical coverage of all three parameters ($\theta_E$, $s_x$, $s_y$) follows the diagonal from 5% to 95% nominal coverage, and the SBC rank histograms (100 test samples) show no systematic over- or under-dispersion.
- **Amortized Inference:** Drawing 500 posterior samples with the cINN takes 0.017 s, about 7× faster than a single Nelder-Mead point estimate on the same image (0.12 s); the ratio varies between runs (7–10×). The Nelder-Mead run starts close to the true parameters, so this is an indicative rather than a like-for-like comparison.
- **Latent-Space OOD Scores:** Observations are scored by $\|z\|^2$, which follows $\chi^2_3$ for in-distribution data under a calibrated model. Each OOD set changes a single factor:

  | OOD set | AUC |
  |---|---|
  | 3× background noise | 0.990 |
  | PSF FWHM doubled (0.16″ → 0.32″) | 0.885 |
  | Einstein radius beyond the prior ($\theta_E \in [2.2, 2.5]$) | 1.000 |

  At the 95% $\chi^2_3$ threshold, the fraction of flagged observations rises from 7.9% at the nominal noise level (5% expected) to 79.7%, 97.1%, 99.1%, and 99.8% at 2×–5× noise.

### 🔧 Fix History
*Post-submission revision (v1.1) by Zhikai Zhang ([@Qck-KK](https://github.com/Qck-KK)). The submitted version is tagged `v1.0-submitted`.*

The first version of the cINN clamped each coupling scale with `tanh`, which bounds the total log-determinant at 9 (3 of the 6 blocks transform 1 dimension and 3 transform 2). Training converged to NLL ≈ −8.96, right at this bound, and the posteriors were much too wide: empirical coverage was already ~40% at 5% nominal, and SBC ranks were concentrated in the middle. The current version standardizes $\theta$ with the prior mean and standard deviation and uses the soft clamp `2·tanh(s/2)`; training now reaches NLL ≈ −16.7 and the posteriors are calibrated. In the same revision, the noise and PSF OOD sets were restricted to the training prior for $\theta_E$ (previously they also shifted $\theta_E$, which made AUC = 1.000 trivial), and a noise-sweep plot that had been generated from random $\chi^2$ samples was replaced with the model's actual scores.

### ⚠️ Known Limitations
- **The OOD score uses the true parameters.** $\|z\|^2$ is computed at the true $\theta$, which is only known in simulation. It is a diagnostic of the learned posterior, not a detector that can be applied directly to real observations.
- **Fixed lens ellipticity in resimulation.** Ellipticity is sampled during training but not stored in the dataset, so the posterior predictive check and the Nelder-Mead baseline use a fixed ellipticity of (0.1, 0.1).

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
| `FinalReport.pdf` | Final project report as submitted. It describes the first version of the model, so its calibration and OOD results (including AUC = 1.000 on all sets) are superseded by the Fix History above. |
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
