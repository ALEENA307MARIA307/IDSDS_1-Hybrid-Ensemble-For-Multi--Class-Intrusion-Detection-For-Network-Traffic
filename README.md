# Hybrid Ensemble for Multi-Class Intrusion Detection in Network Flows

Try the live model here:  
[**Live Demo Space**](https://aleena307maria307-idsds-1.hf.space)  


---

This project implements a **hybrid ensemble model** for multi-class intrusion detection in network traffic.  

The model combines:

- A **blending ensemble** of base models: Decision Tree, Extra Trees, Random Forest  
- Individual classifiers: **XGBoost**, **LightGBM**, **Logistic Regression**  
- A **soft voting mechanism** to produce final predictions

The model leverages predicted probabilities from base models to train the blender and optimize final decisions.

---

## Model Architecture

<img width="621" height="863" alt="idsds_!c drawio" src="https://github.com/user-attachments/assets/22f6c980-c7b7-4cd1-9321-150747df848d" />

---

## Dataset

This project uses the **UNSW-NB15** dataset.  

> The dataset is **not included** in this repository. You can download it from the official source:

[UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---

## Model Registry

Trained model artifacts are hosted on Hugging Face Hub:  
[https://huggingface.co/ALEENA307MARIA307/IDSDS_1](https://huggingface.co/ALEENA307MARIA307/IDSDS_1)

Models are automatically loaded in the deployment via the `huggingface_hub` API.

---

## Accepted Features

<details>
<summary>Click to view accepted features</summary>

See `features.txt` for the full list of input features expected by the model.

</details>
