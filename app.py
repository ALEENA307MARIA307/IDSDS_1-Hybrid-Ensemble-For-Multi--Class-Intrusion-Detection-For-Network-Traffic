
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from ensemble import EnsemblePredictor  

app = FastAPI(
    title="Ensemble Model API",
    description="An API that serves predictions from an ensemble machine learning model",
    version="0.121.0"
)


import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

dt = joblib.load(os.path.join(MODEL_DIR, "dt1.pkl"))
et = joblib.load(os.path.join(MODEL_DIR, "et1.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf1.pkl"))
xg = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))
lr = joblib.load(os.path.join(MODEL_DIR, "lg.pkl"))
lg = joblib.load(os.path.join(MODEL_DIR, "lgb.pkl"))
meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))


attack_labels = {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}

# Initialize ensemble
ensemble = EnsemblePredictor(
    base_models=[dt, et, rf],
    meta_model=meta_model,
    b_models=[xg, lr, lg],
    voting='soft'
)

API_KEY = os.getenv("API_KEY", "default-key")  

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

@app.post("/predict")
def predict(data: PredictionInput, api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        df = pd.DataFrame([data.dict()])
        for col in feature_columns:   
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]  

        pred = ensemble.predict(df)[0]
        attack_status = "Attack" if pred > 0 else "Not an attack"
        attack_label = attack_labels.get(pred, "Unknown")

        return {
            "prediction": int(pred),
            "attack_status": attack_status,
            "attack_label": attack_label
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
