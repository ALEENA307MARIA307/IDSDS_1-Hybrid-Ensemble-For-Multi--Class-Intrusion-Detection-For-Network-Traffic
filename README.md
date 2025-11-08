# Hybrid Ensemble for Multi-Class Intrusion Detection in Network Flows

This project implements a **hybrid ensemble model** for multi-class intrusion detection in network traffic.  

The model combines:

- A **blending ensemble** of base models: Decision Tree, Extra Trees, Random Forest  
- Individual classifiers: **XGBoost**, **LightGBM**, **Logistic regression** and **SVM**  
- A **soft voting mechanism** to produce final predictions

The model leverages predicted probabilities from base models to train the blender and optimize final decisions.

---

## Model Architecture

<img width="621" height="863" alt="idsds_!c drawio" src="https://github.com/user-attachments/assets/22f6c980-c7b7-4cd1-9321-150747df848d" />





---

## Dataset

This project uses the **UNSW-NB15** dataset.  

The dataset is **not included** in this repository. You can download it from the official source here:  

[UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---



