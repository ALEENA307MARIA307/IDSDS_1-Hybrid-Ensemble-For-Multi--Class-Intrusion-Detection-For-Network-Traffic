import numpy as np
import pandas as pd

class EnsemblePredictor:
    def __init__(self, base_models, meta_model, b_models, voting='soft'):
        self.base_models = base_models
        self.meta_model = meta_model
        self.voting = voting
        self.b_models = b_models

    def predict(self, X, X_m):
      
        # Base models for meta model predict on X_m
        base_probs = np.hstack([m.predict_proba(X_m) for m in self.base_models])
        
        # Meta model predicts based on base_probs
        meta_prob = self.meta_model.predict_proba(base_probs)[:, 1]

        if self.voting == 'soft':
            # Blending models predict on X
            b_probs = np.hstack([m.predict_proba(X) for m in self.b_models])
            
            # Combine meta model and blending model outputs
            all_probs = np.hstack([meta_prob.reshape(-1, 1), b_probs])
            
            # Average probabilities
            final_probs = np.mean(all_probs, axis=1)
            return (final_probs >= 0.5).astype(int)
        else:
            return (meta_prob >= 0.5).astype(int)
