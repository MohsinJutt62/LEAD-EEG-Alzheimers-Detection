# 🧠 LEAD-EEG: Lightweight Ensemble for Alzheimer’s Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-76.14%25-success)

## 📌 Project Overview
This repository contains the official implementation of the **LEAD-EEG** framework. It is an AI-based diagnostic tool designed to automatically classify Alzheimer's Disease (Dementia) vs. Healthy Controls using raw brain electrical signals (EEG). 

The major challenge in medical data processing is memory limitations. This framework is highly hardware-optimized, utilizing MNE lazy-loading and mathematical feature expansion (extracting 125 complex features from 4 foundational signals) to completely eliminate Out-of-Memory (OOM) crashes on standard machines.

##Images
<img width="1395" height="755" alt="Ensemble (Random Forest aur Gradient Boosting)" src="https://github.com/user-attachments/assets/4df837d8-90a6-41c6-8eab-537b9d237a61" />

## 📊 Dataset Specifications
* **Source:** OpenNeuro (ds004504)
* **Total Subjects:** 176 Patients 
* **Class Distribution:** 88 Alzheimer's Patients + 88 Healthy Controls
* **Format:** `.set` EEG files

## 🚀 Key Features & Architecture
1. **RAM-Safe Loading:** Lazy loading and strict signal cropping (first 20,000 samples).
2. **Feature Engineering:** Signal Energy, Power, and Spectral Entropy expanded via Degree-4 Polynomial Expansion.
3. **Data Balancing:** Integrated SMOTE within a 10-Fold Stratified Cross-Validation loop.
4. **Ensemble Classifier:** A Soft Voting architecture combining **Random Forest** and **Gradient Boosting**.

*(You can upload your architecture diagram image here in the repo and link it)*

## 🏆 Final Classification Results (10-Fold CV)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Dementia (AD)** | 0.81 | 0.84 | 0.82 | 118 |
| **Healthy Control** | 0.65 | 0.60 | 0.62 | 58 |
| **Overall Accuracy** | - | - | **76.14%** | 176 |

## ⚙️ How to Run on Your Machine
1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/LEAD-EEG-Alzheimers-Detection.git](https://github.com/YourUsername/LEAD-EEG-Alzheimers-Detection.git)
