import torch
import torchvision.transforms as transforms
from pathlib import Path


class Config:
    def __init__(self):
        """Initialize configuration settings"""
        self.image_size = 224  # Standard size for many CNN architectures
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]   # ImageNet std
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Package directory for asset paths
        self.package_dir = Path(__file__).parent.parent
        
        # Model paths
        self.model_path = self.package_dir / "trainer_model.pth"
        self.class_names_path = self.package_dir / "class_names.txt"
    
    def get_device(self) -> str:
        """Get the device to use for computations"""
        return self.device
    
    def get_val_transform(self) -> transforms.Compose:
        """
        Get transformation pipeline for validation/inference
        
        Returns:
            Composed transformation pipeline
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
