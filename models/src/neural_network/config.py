from pathlib import Path
from typing import Any

import torchvision.transforms as transforms

# Get current directory using pathlib
CURRENT_DIR = Path(__file__).parent.absolute()


class Config:
    # Training Configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    IMAGE_SIZE = 224
    
    # Dataset Configuration
    DATASET_PATH = CURRENT_DIR / 'dataset' / 'images'
    TRAIN_VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Model Configuration
    MODEL_SAVE_PATH = CURRENT_DIR / 'trainers_model.pth'
    CLASS_NAMES_PATH = CURRENT_DIR / 'class_names.json'

    # Normalization values for ImageNet
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Hardware Configuration
    @staticmethod
    def get_device() -> Any:
        import torch
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def _get_val_transform(self) -> transforms.Compose:
        """
        Get validation/inference transforms for image preprocessing
        
        Returns:
            torchvision.transforms.Compose: Composed transform pipeline
        """
        return transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.NORMALIZE_MEAN,
                std=self.NORMALIZE_STD
            )
        ])

    # Optional: Helper method to create directories if they don't exist
    @classmethod
    def create_directories(cls) -> None:
        cls.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
