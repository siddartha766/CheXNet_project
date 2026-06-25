# CheXNet_project
# Clinical Pneumonia Detection via Transfer Learning and XAI

## Project Overview
Welcome to my Pneumonia Detection project. This repository contains the codebase for a deep learning diagnostic tool built to detect pneumonia from frontal-view chest X-ray images. 

My primary focus for this project wasn't just to achieve strong quantitative metrics (like AUROC), but to build a clinically trustworthy model. To ensure the AI was making decisions based on actual pathology rather than background noise, I heavily integrated Explainable AI (XAI) into the evaluation pipeline.

## Dataset Context
* The data used in this project is a targeted subset derived from the **NIH ChestX-ray14 Dataset**[cite: 14]. 
* The original dataset contains over 112,000 frontal-view X-ray images of 30,805 patients extracted from the clinical  database at the National Institutes of Health[cite: 9, 13].
* While the full dataset includes 14 different thoracic pathologies[cite: 14], this project specifically isolates the **Pneumonia** diagnostic track to optimize the model's classification head for a single, critical disease.

## Methodology & Training Strategy
The architecture utilizes a pre-trained convolutional neural network (DenseNet-121) adapted for medical imaging. To ensure stable learning, I implemented a two-phase training pipeline:

* **Phase 1 (Feature Extraction):** The deep convolutional layers were frozen to preserve the pre-trained edge-detection weights. Only the fully connected classification head was trained during this phase to establish a baseline.
* **Phase 2 (Deep Fine-Tuning):** The entire network was unfrozen to allow for deep feature adaptation. To prevent "catastrophic forgetting" of the valuable pre-trained weights, the learning rate was strictly decayed to `0.00001`. This conservative optimization ensured smooth convergence.

## Model Interpretability & Clinical Validation
A core component of this pipeline is the use of **Grad-CAM** (Gradient-weighted Class Activation Mapping) to audit the model's spatial attention. 

In clinical AI, high accuracy scores can sometimes hide data leakage or "shortcut learning." By generating Grad-CAM heatmaps for the test images, I was able to successfully validate that the model correctly targets pulmonary opacities within the thoracic cavity. The visual attention maps align closely with true anatomical markers, confirming that the model has learned the actual clinical features of pneumonia rather than relying on dataset artifacts.

## Current Constraints & Future Scope
1. **Computational Limits:** Due to immediate compute and time constraints, the Phase 2 fine-tuning was done until 4 epochs. While this successfully proved the core pipeline and yielded strong Grad-CAM results, deeper convergence (50+ epochs) is planned for future iterations.
2. **Infrastructure Upgrades**:Future work will focus on extending training for higher precision and code modifications.
