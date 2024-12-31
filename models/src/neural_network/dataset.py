import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Any


class DogDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None) -> Any:
        self.image_dir = image_dir
        self.transform = transform
        self.data_pairs = []
        self.classes = []
        self._setup_dataset()
    
    def _setup_dataset(self) -> Any:
        """Setup dataset structure and class mapping"""
        self.classes = [d.name for d in self.image_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        self.class_to_idx = {cls_name: idx for idx, cls_name 
                            in enumerate(sorted(self.classes))}
        
        for class_name in self.classes:
            img_class_dir = self.image_dir / class_name
            if img_class_dir.exists():
                for img_path in img_class_dir.glob('*.jpg'):
                    self.data_pairs.append({
                        'image': img_path,
                        'class': class_name
                    })
        
        self._validate_dataset()
    
    def _validate_dataset(self) -> Any:
        """Validate dataset integrity"""
        if len(self.data_pairs) == 0:
            raise ValueError("No images found in dataset!")
        print(f"Found {len(self.data_pairs)} images across {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx) -> Any:
        data_pair = self.data_pairs[idx]
        image = Image.open(data_pair['image']).convert('RGB')
        label = self.class_to_idx[data_pair['class']]
        
        if self.transform:
            image = self.transform(image)
        return image, label

class DatasetManager:
    def __init__(self, config) -> Any:
        self.config = config
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self) -> transforms:
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _get_val_transform(self) -> transforms:
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_data_loaders(self) -> Any:
        # Setup dataset
        full_dataset = DogDataset(self.config.DATASET_PATH, transform=None)
        
        # Split indices
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=self.config.TRAIN_VAL_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=[pair['class'] for pair in full_dataset.data_pairs]
        )
        
        # Create datasets
        train_dataset = self._create_split_dataset(full_dataset, train_idx, self.train_transform)
        val_dataset = self._create_split_dataset(full_dataset, val_idx, self.val_transform)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, len(full_dataset.classes)
    
    def _create_split_dataset(self, full_dataset, indices, transform) -> DogDataset:
        dataset = DogDataset(self.config.DATASET_PATH, transform=transform)
        dataset.data_pairs = [full_dataset.data_pairs[i] for i in indices]
        dataset.classes = full_dataset.classes
        dataset.class_to_idx = full_dataset.class_to_idx
        return dataset
