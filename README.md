# Hybrid Ensemble for Multi-Class Intrusion Detection in Network Flows

This project implements a **hybrid ensemble model** for multi-class intrusion detection in network traffic.  

The model combines:

- A **blending ensemble** of base models: Decision Tree, Extra Trees, Random Forest  
- Individual classifiers: **XGBoost**, **LightGBM**, **Logistic regression**  
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

## Model Registry

The trained model artifacts are hosted on the Hugging Face Hub:
[https://huggingface.co/ALEENA307MARIA307/IDSDS_1](https://huggingface.co/ALEENA307MARIA307/IDSDS_1)

These models are automatically loaded in the deployment via the `huggingface_hub` API.

---

## Live Demo

You can try the model live here:

[Try the live Space](https://ALEENA307MARIA307-IDSDS_1.hf.space)

Visit `/docs` on the Space URL to see the interactive API documentation.

---

## Features Accepted

The model expects the following input features:

| Feature Name       | Type  | Description |
|-------------------|-------|-------------|
| sttl               | float | Source-to-destination **TTL** (time-to-live) value of the packet. |
| smean              | float | Mean packet **size from the source**. |
| ct_dst_src_ltm     | float | Count of connections from **source to destination in the last time window**. |
| ct_state_ttl       | float | Count of connections in a specific **state with similar TTL**. |
| ct_srv_src         | float | Count of connections to the **same service from the same source**. |
| sload              | float | Source **load**, i.e., total bytes sent by source over a time window. |
| tcprtt             | float | **Round-trip time** of TCP packets between source and destination. |
| dmean              | float | Mean packet **size from the destination**. |
| service            | float | Encoded **service type** (e.g., HTTP, FTP, DNS). |
| rate               | float | Rate of **packets per second** from the source. |
| dload              | float | Destination **load**, i.e., total bytes received by destination. |
| dinpkt             | float | Number of **incoming packets** to the destination. |
| dttl               | float | Destination packet **TTL** (time-to-live). |
| dur                | float | Duration of the connection/session. |
| sjit               | float | **Jitter of the source** packets (variation in packet arrival time). |
| sinpkt             | float | Number of **incoming packets** from the source. |
| djit               | float | **Jitter of the destination** packets. |
| dpkts              | float | Total number of packets sent by the destination. |
| ct_dst_ltm         | float | Count of connections to the **destination in the last time window**. |
| spkts              | float | Total number of packets sent by the source. |


>  Make sure your input JSON matches these features when using the `/predict` endpoint.

---

