# explainability/saliency.py

import torch
from captum.attr import Saliency


class SaliencyExplainer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.saliency = Saliency(self.model)

    def explain(self, state_tensor, target_action: int):
        """
        state_tensor: 1D torch tensor (state) on correct device
        target_action: int, index of the action you want to explain
        Returns: 1D saliency values (array) for input features
        """
        state_tensor = state_tensor.unsqueeze(
            0
        ).requires_grad_()  # Shape: [1, input_dim]
        attr = self.saliency.attribute(state_tensor, target=target_action)
        return attr.squeeze().detach().cpu().numpy()
