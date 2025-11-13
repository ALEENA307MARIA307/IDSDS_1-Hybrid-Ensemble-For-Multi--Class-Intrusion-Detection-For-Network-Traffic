import json
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from ensemble import EnsemblePredictor
from huggingface_hub import hf_hub_download
import gradio as gr

# FastAPI app
app = FastAPI(
    title="Hybrid Ensemble IDS API",
    description="JSON input -> encoded, scaled -> hybrid ensemble prediction",
    version="1.0.0"
)

# Hugging Face repo
HF_REPO = "ALEENA307MARIA307/IDSDS_1"

# Load models/artifacts
dt = joblib.load(hf_hub_download(HF_REPO, "dt1.pkl"))
et = joblib.load(hf_hub_download(HF_REPO, "et1.pkl"))
rf = joblib.load(hf_hub_download(HF_REPO, "rf1.pkl"))
xg = joblib.load(hf_hub_download(HF_REPO, "xgb.pkl"))
lr = joblib.load(hf_hub_download(HF_REPO, "lg.pkl"))
lg = joblib.load(hf_hub_download(HF_REPO, "lgb.pkl"))
meta_model = joblib.load(hf_hub_download(HF_REPO, "meta_model.pkl"))

scaler = joblib.load(hf_hub_download(HF_REPO, "scaler.pkl"))
scaler_meta = joblib.load(hf_hub_download(HF_REPO, "scaler_meta.pkl"))
feature_columns = joblib.load(hf_hub_download(HF_REPO, "selected_features.pkl"))
encoders = joblib.load(hf_hub_download(HF_REPO, "encoders.pkl"))

attack_labels = {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}

ensemble = EnsemblePredictor(
    base_models=[dt, et, rf],
    meta_model=meta_model,
    b_models=[xg, lr, lg],
    scaler_meta=scaler_meta,
    voting='soft'
)

# Encode categorical columns
def encode_categoricals(df, encoders):
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df.columns:
            le = encoders[col]
            df[col] = df[col].map(lambda s: s if s in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            df[col] = le.transform(df[col])
    return df

# Prepare input up to feature selection/reindex
def preprocess_step1(input_dict):
    df = pd.DataFrame([input_dict])
    df = encode_categoricals(df, encoders)

    # Scale only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Keep only selected features
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# Apply meta scaling
# def preprocess_step2(df):
#     numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     c = scaler_meta.transform(df[numerical_cols])
#     return c

# Gradio pipeline
def gradio_predict(raw_input):
    try:
        # Parse the input as JSON
        input_dict = json.loads(raw_input)
    except Exception as e:
        return {"error": f"Invalid JSON input: {str(e)}"}

    # Preprocess
    a = preprocess_step1(input_dict)
    #b = preprocess_step2(a)

    # Predict
    pred = ensemble.predict(a, a)[0]
    label = attack_labels.get(pred, "Unknown")
    return {
        "prediction": int(pred),
        "attack_label": label,
        "alert": "Intrusion detected!" if label != "Normal" else "Normal traffic"
    }

# Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Input dictionary features",
        placeholder='Enter JSON dictionary, e.g. {"sttl": 1.0, "smean": 0.5, ...}',
        lines=10
    ),
    outputs=gr.JSON(label="Prediction output"),
    title="Hybrid Ensemble IDS",
    description="Provide a dictionary in JSON format with network features; the model will preprocess, scale, and predict.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)