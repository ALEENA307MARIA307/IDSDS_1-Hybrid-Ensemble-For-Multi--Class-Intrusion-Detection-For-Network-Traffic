from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from ensemble import EnsemblePredictor
from huggingface_hub import hf_hub_download

# Initialize API
app = FastAPI(
    title="Hybrid Ensemble Intrusion Detection API",
    description="An API serving predictions from a hybrid ensemble model for multi-class intrusion detection.",
    version="1.0.0"
)

# Hugging Face repo
HF_REPO = "ALEENA307MARIA307/IDSDS_1"

# --- Load models and artifacts from Hugging Face Hub ---
dt = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="dt1.pkl"))
et = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="et1.pkl"))
rf = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="rf1.pkl"))
xg = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="xgb.pkl"))
lr = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="lg.pkl"))
lg = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="lgb.pkl"))
meta_model = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="meta_model.pkl"))

# Load scalers
scaler = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="scaler.pkl"))
scaler_meta = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="scaler_meta.pkl"))

# Load features
feature_columns = joblib.load(hf_hub_download(repo_id=HF_REPO, filename="selected_features.pkl"))

# Attack labels for decoding predictions
attack_labels = {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}

# Initialize ensemble predictor
ensemble = EnsemblePredictor(
    base_models=[dt, et, rf],
    meta_model=meta_model,
    b_models=[xg, lr, lg],
    voting='soft'
)

# API Key Authentication
API_KEY = os.getenv("API_KEY")


# Input Schema
class PredictionInput(BaseModel):
    sttl: float
    smean: float
    ct_dst_src_ltm: float
    ct_state_ttl: float
    ct_srv_src: float
    sload: float
    tcprtt: float
    dmean: float
    service: float
    rate: float
    dload: float
    dinpkt: float
    dttl: float
    dur: float
    sjit: float
    sinpkt: float
    djit: float
    dpkts: float
    ct_dst_ltm: float
    spkts: float



@app.get("/features")
def get_features():
    """Return expected input features for the model."""
    return {"expected_features": feature_columns}


@app.post("/predict")
def predict(data: PredictionInput, api_key: str = Header(...)):
    """Generate a prediction from the ensemble model."""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Add missing columns (if any)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        # Scale inputs
        X_scaled = scaler.transform(df)
        X_meta_scaled = scaler_meta.transform(df)

        # Predict using ensemble
        pred = ensemble.predict(X_scaled, X_meta_scaled)[0]

        return {
            "prediction": int(pred),
            "attack_label": attack_labels.get(pred, "Unknown"),
            "description": "Predicted network traffic class based on hybrid ensemble model."
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

