# CheXNet_project
# Clinical Pneumonia Detection via Transfer Learning and XAI

## Project Overview

* This project focuses on building a pneumonia detection model that is not only accurate but also clinically trustworthy. Rather than focusing solely on quantitative metrics like AUROC, the project prioritizes Explainable AI (XAI) to verify that the model’s decisions are based on actual clinical pathology rather than image artifacts. The project utilizes the PyTorch framework to develop and DenseNet-121 architecture to detect pneumonia from chest X-rays. 
* This repository contains the codebase for a deep learning diagnostic tool built to detect pneumonia from frontal-view chest X-ray images.
  
## Dataset Context
* The data used in this project is a targeted subset derived from the **NIH ChestX-ray14 Dataset**. 
* The original dataset contains over 112,000 frontal-view X-ray images of 30,805 patients extracted from the clinical  database at the National Institutes of Health.
* While the full dataset includes 14 different thoracic pathologies, this project specifically isolates the **Pneumonia** diagnostic track to optimize the model's classification head for a single, critical disease.

## Methodology & Training Strategy
The architecture utilizes a pre-trained convolutional neural network (DenseNet-121) adapted for medical imaging. To ensure stable learning, I implemented a two-phase training pipeline:

* **Phase 1 (Feature Extraction):** The deep convolutional layers were frozen to preserve the pre-trained edge-detection weights. Only the fully connected classification head was trained during this phase to establish a baseline.
* **Phase 2 (Deep Fine-Tuning):** The entire network was unfrozen to allow for deep feature adaptation. To prevent "catastrophic forgetting" of the valuable pre-trained weights, the learning rate was strictly decayed to `0.00001`. This conservative optimization ensured smooth convergence.

## Model Interpretability & Clinical Validation
A core component of this pipeline is the use of **Grad-CAM** (Gradient-weighted Class Activation Mapping) to audit the model's spatial attention. 

In clinical AI, high accuracy scores can sometimes hide data leakage or "shortcut learning." By generating Grad-CAM heatmaps for the test images, I was able to successfully validate that the model correctly targets pulmonary opacities within the thoracic cavity. The visual attention maps align closely with true anatomical markers, confirming that the model has learned the actual clinical features of pneumonia rather than relying on dataset artifacts.

## Development Methodology and References
* This implementation is based on the methodology established in the CheXNet framework Research paper (Rajpurkar et al., 2017).
  
* AI Tools were used for assistance


## Current Constraints & Future Scope
1. **Computational Limits:** Due to immediate compute and time constraints, the Phase 2 fine-tuning was done until 4 epochs. While this successfully proved the core pipeline and yielded strong Grad-CAM results, deeper convergence (50+ epochs) is planned for future iterations.
2. **Infrastructure Upgrades**:Future work will focus on extending training for higher precision and code modifications.

## Repository Structure & File Guide
To facilitate future development and reproducibility, the codebase is modularized. Here is a guide to navigating the scripts:

* `cell_1.py` - [Environment initialization, core dependency imports, and CUDA/GPU hardware acceleration configuration.]
* `cell_2.py` - [Environment verification and directory mapping]
* `cell_3.py` - [Dataset indexing, binary target extraction, and strict patient-level dataset splitting (Train/Val/Test) to prevent clinical data leakage.]
* `cell_4.py` - [Image preprocessing and robust data augmentation pipelines to standardize inputs and prevent model overfitting.]
* `cell_5.py` - [Custom PyTorch Dataset and DataLoader classes to handle batching, transformations, and memory-efficient tensor feeding to the GPU.]
* `cell_6.py` - [Data pipeline verification,and visual sanity checking to validate the augmentation process.]
* `cell_7.py` - [DenseNet-121 Transfer Learning with Custom Pneumonia binary Classification Head.]
* `cell_8.py` - [Training Configuration: Weighted BCE Loss Implementation for Class Imbalance and Adam Optimizer Setup.]
* `cell_9.py` - [Master Training Loop: Model training, validation, dynamic LR scheduling, and checkpoint-based fault tolerance.]
* `cell_10.py` - [Model Evaluation: Final clinical performance assessment, AUROC computation, and confusion matrix visualization.]
* `cell_11.py` - [Model Explainability: Implementation of Grad-CAM, Guided Backpropagation, and Guided Grad-CAM for clinical visualization.]
