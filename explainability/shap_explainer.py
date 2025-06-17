# explainability/shap_explainer.py

import shap
import numpy as np
import torch


class SHAPExplainer:
    def __init__(self, model, input_dim, device="cpu"):
        """
        model: a function that maps np.array -> Q-values
        input_dim: size of observation space
        """
        self.input_dim = input_dim
        self.device = device

        def model_wrapper(x_np):
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return model(x_tensor).cpu().numpy()

        self.explainer = shap.Explainer(
            model_wrapper, shap.maskers.Independent(np.zeros((1, input_dim)))
        )

    def explain(self, state):
        """
        state: np.array of shape (input_dim,)
        Returns: SHAP values for each input dimension
        """
        return self.explainer(np.array([state]))
