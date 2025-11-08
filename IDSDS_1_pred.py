# LOAD objects

import joblib
dt=joblib.load("dt1.pkl")
et=joblib.load("et1.pkl")
rf=joblib.load("rf1.pkl")
xg=joblib.load("xgb.pkl")
#svm=joblib.load("svm.pkl")
lr=joblib.load("lg.pkl")
lg=joblib.load("lgb.pkl")
scaler=joblib.load("scaler.pkl")
meta_model=joblib.load("meta_model.pkl")
encoder=joblib.load("encoders.pkl")
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import lightgbm
import xgboost
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import mlclassifier 


import numpy as np

class ensemble_pred:
    def __init__(self, base_models, meta_model, b_models, voting='soft'):
        self.base_models = base_models
        self.meta_model = meta_model
        self.voting = voting
        self.b_models=b_models

    def predict(self, X):
        
        X = encoder.transform(X)
        X = scaler.transform(X)

        # Get predictions from all base models
        base_probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models])
        meta_input = np.hstack(base_probs)

        # Get meta-learner prediction
        meta_input = base_probs
        #meta_pred = self.meta_model.predict(meta_input)
        meta_prob = self.meta_model.predict_proba(meta_input)[:, 1]

        # Combine using your logic
        if self.voting == 'soft':
            # average probabilities of base models + meta learner
            b_probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.b_models])
            meta_input = np.hstack(b_probs)

            all_probs = np.column_stack(base_probs + [meta_prob] + b_probs)
            final_probs = np.mean(all_probs, axis=1)
            return (final_probs >= 0.5).astype(int)
    


ensemble = ensemble_pred(
    base_models=[dt,et,rf],
    meta_model=meta_model,
    b_models=[xg,lr,lg],
    voting='hard'
)



joblib.dump(ensemble,"ensemble_1.pkl")



