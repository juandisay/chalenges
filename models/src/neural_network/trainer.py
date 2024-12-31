import torch
import json
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any


class ModelTrainer:
    def __init__(self, model, config) -> Any:
        self.model = model
        self.config = config
        self.device = config.get_device()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = self._get_scheduler()
        self.class_names = None  # Will store class names

    def _get_scheduler(self) -> Any:
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )

    def save_class_names(self, dataset) -> None:
        """Save class names and their indices to a JSON file"""
        class_mapping = {
            'class_names': dataset.classes,
            'class_to_idx': dataset.class_to_idx
        }
        
        with open(self.config.CLASS_NAMES_PATH, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        
        print(f"Class names saved to {self.config.CLASS_NAMES_PATH}")
    
    def load_class_names(self) -> dict:
        """Load class names and their indices from JSON file"""
        try:
            with open(self.config.CLASS_NAMES_PATH, 'r') as f:
                class_mapping = json.load(f)
            return class_mapping
        except FileNotFoundError:
            print("No class mapping file found!")
            return None
    
    def train(self, train_loader, val_loader) -> dict:
        # Save class names at the start of training
        self.save_class_names(train_loader.dataset)
        
        # Rest of the training code remains the same
        best_val_acc = 0.0
        history = {
            'train_losses': [], 'train_accs': [],
            'val_losses': [], 'val_accs': []
        }
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            history['val_losses'].append(val_loss)
            history['val_accs'].append(val_acc)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Print progress
            self._print_epoch_progress(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model()
        
        return history
    
    def _train_epoch(self, train_loader) -> Any:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            loss, acc = self._train_step(images, labels)
            running_loss += loss
            correct += acc[0]
            total += acc[1]
            
            if (i + 1) % 10 == 0:
                print(f'Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}')
        
        return running_loss / len(train_loader), 100 * correct / total
    
    def _train_step(self, images, labels) -> Any:
        images, labels = images.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        
        return loss.item(), (correct, total)
    
    def _validate(self, val_loader) -> Any:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return val_loss / len(val_loader), 100 * correct / total
    
    def _save_model(self) -> torch:
        torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
    
    def _print_epoch_progress(self, epoch, train_loss, train_acc, val_loss, val_acc):
        print(f'Epoch [{epoch+1}/{self.config.NUM_EPOCHS}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    def plot_training_history(self, history) -> plt:
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accs'], label='Training Accuracy')
        plt.plot(history['val_accs'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def predict_image(self, image_path: str) -> tuple:
        """
        Predict the class of a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (predicted_class_name, confidence)
        """
        # Load class names
        class_mapping = self.load_class_names()
        if not class_mapping:
            raise ValueError("No class mapping found! Please train the model first.")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = self.config._get_val_transform()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Get class name
        predicted_idx = predicted.item()
        class_names = class_mapping['class_names']
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, confidence.item()
