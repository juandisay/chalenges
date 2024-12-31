import torch

import torch.nn as nn
import torchvision.models as models
from typing import Any


class DogClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(DogClassifier, self).__init__()
        # Load pre-trained model
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self._freeze_base_layers()
        self._setup_classifier(num_classes)
    
    def _freeze_base_layers(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _setup_classifier(self, num_classes: int) -> None:
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.model.last_channel, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def load_state_dict(self, state_dict: dict) -> Any:
        """Custom state dict loading to handle model prefix"""
        # Remove 'model.' prefix from state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # Load the modified state dict
        return super().load_state_dict(new_state_dict, strict=False)
