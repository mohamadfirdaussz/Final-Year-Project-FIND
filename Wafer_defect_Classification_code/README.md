# WM-811K Wafer Defect Classification

This project implements an automated machine learning pipeline for classifying defects in semiconducting wafers using the WM-811K (LSWMD) dataset.

## 🚀 Quick Start (One-Click Run)

The easiest way to run the entire pipeline (environment setup, dependency installation, and execution) is using the `run_all.py` script.

1.  **Ensure you have Python 3.10 - 3.12 installed.**
2.  **Download the dataset**:
    *   Download `LSWMD.pkl` from [Kaggle: WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
    *   Place it in: `ml_flow/datasets/LSWMD.pkl`.
3.  **Run the script**:
    ```powershell
    python ml_flow/run_all.py
    ```

---

## 🛠️ Pipeline Architecture

The pipeline is structured into 5 sequential stages, managed by `ml_flow/main.py`:

1.  **Stage 1: Data Loading & Cleaning** (`data_loader.py`)
    *   Cleans labels and removes wafers smaller than 5x5.
    *   Applies median filtering for denoising.
    *   Resizes all wafers to 64x64.
2.  **Stage 2: Feature Engineering** (`feature_engineering.py`)
    *   Extracts 66 domain-specific features (Density, Radon transform, Geometry).
3.  **Stage 3: Preprocessing** (`data_preprocessor.py`)
    *   Performs stratified 80/20 train/test split.
    *   Applies standard scaling and hybrid balancing (SMOTE + Undersampling).
4.  **Stage 3.5: Feature Expansion** (`feature_combination.py`)
    *   Generates interaction terms, expanding to ~6,500 features.
5.  **Stage 4: Feature Selection** (`feature_selection.py`)
    *   Reduces dimensionality via 3 parallel tracks: ANOVA + RFE, Random Forest, and Lasso.
6.  **Stage 5: Model Training & Evaluation** (`model_tuning.py`)
    *   Trains 7 different models per track with 3-fold cross-validation.
    *   Saves the best models and comprehensive evaluation metrics.

---

## 📂 Project Structure

*   `ml_flow/`: Contains all source code for the pipeline.
    *   `run_all.py`: One-click entry point (manages venv and dependencies).
    *   `main.py`: Pipeline orchestrator.
    *   `config.py`: Centralized path and logging configuration.
    *   `datasets/`: Directory for input data (`LSWMD.pkl`).
*   `data_loader_results/`: Cleaned wafer maps.
*   `Feature_engineering_results/`: Extracted base features.
*   `preprocessing_results/`: Balanced and scaled data.
*   `feature_selection_results/`: Data for different feature tracks.
*   `model_artifacts/`: Trained models and `master_model_comparison.csv`.

## 📋 Requirements

*   **Python**: 3.10 to 3.12 (Version 3.13+ is not yet supported).
*   **Operating System**: Windows (tested) / Linux / macOS.
*   **Dependencies**: Automatically installed by `run_all.py` into a local `.venv`.

---
*Developed for Academic Machine Learning Project.*
