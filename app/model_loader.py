# app/model_loader.py
import torch
from torchvision import models

class ModelLoader:
    def __init__(self, model_name: str, weights_path: str, device: str = "cpu"):
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = device
        self.model = self._create_model()

    def _create_model(self):
        if self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return model

    def load_weights(self):
        state = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        return self.model
