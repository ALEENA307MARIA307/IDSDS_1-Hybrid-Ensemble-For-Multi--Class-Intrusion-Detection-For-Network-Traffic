import numpy as np
import pandas as pd

class EnsemblePredictor:
    def __init__(self, base_models, meta_model, b_models, voting='soft', scaler_meta=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.b_models = b_models
        self.voting = voting
        self.scaler_meta = scaler_meta  # Add scaler_meta

    def predict(self, X, X_m):
        # Base models predict on X_m
        prob_dfs = []
        for i, model in enumerate(self.base_models, start=1):
            probs = model.predict_proba(X)
            n_classes = probs.shape[1]
            col_names = [f'model{i}_class{j}' for j in range(n_classes)]
            prob_dfs.append(pd.DataFrame(probs, columns=col_names))

        # Concatenate all base model probabilities
        base_probs_df = pd.concat(prob_dfs, axis=1)

        # Apply meta scaler if provided
        if self.scaler_meta is not None:
            #numerical_cols = base_probs_df.select_dtypes(include=[np.number]).columns
            base_probs_scaled = base_probs_df.copy()
            base_probs_scaled = self.scaler_meta.transform(base_probs_df)
        else:
            base_probs_scaled = base_probs_df.values

        # Meta model predicts using scaled base model probabilities
        meta_prob = self.meta_model.predict_proba(base_probs_scaled)

        if self.voting == 'soft':
            # Blending models predict on X
            # Blending models predict probabilities on X
            b_probs = np.hstack([m.predict_proba(X) for m in self.b_models])

            # Combine all probabilities (meta + blending)
            n_classes = meta_prob.shape[1]
            all_probs = np.hstack([meta_prob, b_probs])

            # Reshape and average probabilities across models
            all_probs = all_probs.reshape(meta_prob.shape[0], -1, n_classes)
            final_probs = np.mean(all_probs, axis=1)

            # Pick class with highest average probability
            return np.argmax(final_probs, axis=1)
        else:
            pass
