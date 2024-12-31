from dataclasses import dataclass
from typing import List, Dict, Union, Optional, TypeVar, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import logging
from enum import Enum
from neural_network.model import DogClassifier
from neural_network.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type hints
T = TypeVar('T', bound='DogPredictor')
ImageType = Union[str, Path]

@dataclass
class PredictionResult:
    """Data class for prediction results"""
    breed: str
    confidence: float

class ImageFormat(Enum):
    """Supported image formats"""
    JPG = '.jpg'
    JPEG = '.jpeg'
    PNG = '.png'

class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

class ModelLoader:
    """Handles model and class name loading"""
    
    @staticmethod
    def load_class_names(config: Config) -> List[str]:
        """Load class names from saved JSON file"""
        try:
            with open(config.CLASS_NAMES_PATH, 'r') as f:
                data = json.load(f)
                return data['class_names']
        except FileNotFoundError:
            raise FileNotFoundError("Class names file not found! Please train the model first.")
        except json.JSONDecodeError:
            raise ValueError("Invalid class names file format")

    @staticmethod
    def load_model(config: Config, num_classes: int) -> nn.Module:
        """Load the trained model"""
        try:
            model = DogClassifier(num_classes)
            model.load_state_dict(torch.load(
                config.MODEL_SAVE_PATH,
                map_location=config.get_device()
            ))
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

class Visualizer:
    """Handles visualization of predictions"""
    
    @staticmethod
    def plot_predictions(
        image: Image.Image,
        predictions: List[PredictionResult],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot image with predictions"""
        plt.figure(figsize=(12, 6))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Input Image')
        
        # Plot predictions
        plt.subplot(1, 2, 2)
        breeds = [pred.breed for pred in predictions]
        confidences = [pred.confidence for pred in predictions]
        y_pos = range(len(breeds))
        
        plt.barh(y_pos, confidences)
        plt.yticks(y_pos, breeds)
        plt.xlabel('Confidence (%)')
        plt.title('Top Predictions')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class DogPredictor:
    def __init__(self, config: Config) -> None:
        """Initialize predictor with config"""
        self.config = config
        self.device = config.get_device()
        
        # Load class names and model
        self.class_names = ModelLoader.load_class_names(config)
        self.model = ModelLoader.load_model(config, len(self.class_names))
        self.model = self._setup_model(self.model)
        
        logger.info(f"Predictor initialized with {len(self.class_names)} classes")
        logger.info(f"Using device: {self.device}")
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for inference"""
        model = model.to(self.device)
        model.eval()
        return model
    
    def _validate_image_path(self, image_path: ImageType) -> Path:
        """Validate image path and format"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        if path.suffix.lower() not in {format.value for format in ImageFormat}:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        return path
    
    def _load_and_transform_image(self, image_path: Path) -> Tuple[torch.Tensor, Image.Image]:
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            transform = self.config._get_val_transform()
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            return image_tensor, image
        except Exception as e:
            raise ImageProcessingError(f"Error processing image {image_path}: {str(e)}")
    
    def _get_predictions(
        self,
        outputs: torch.Tensor,
        top_k: int = 5
    ) -> List[PredictionResult]:
        """Get top-k predictions"""
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            pred = PredictionResult(
                breed=self.class_names[idx],
                confidence=round(float(prob) * 100, 2)
            )
            predictions.append(pred)
        
        return predictions

    def predict(
        self,
        image_path: ImageType,
        visualize: bool = True,
        save_path: Optional[Path] = None
    ) -> Dict:
        """Predict dog breed for single image with optional visualization"""
        try:
            # Validate and process image
            path = self._validate_image_path(image_path)
            image_tensor, original_image = self._load_and_transform_image(path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                predictions = self._get_predictions(outputs)
            
            # Format result
            result = {
                'filename': path.name,
                'predictions': [pred.__dict__ for pred in predictions]
            }
            
            # Visualize if requested
            if visualize:
                Visualizer.plot_predictions(
                    original_image,
                    predictions,
                    save_path
                )
            
            logger.info(f"Successfully predicted {path.name}")
            return result
            
        except (FileNotFoundError, ValueError, ImageProcessingError) as e:
            raise e
        except Exception as e:
            raise PredictionError(f"Error predicting {path.name}: {str(e)}")

    def predict_batch(
        self,
        image_dir: ImageType,
        output_file: Optional[Path] = None,
        visualize: bool = True
    ) -> Dict:
        """Predict dog breeds for all images in directory"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise NotADirectoryError(f"Directory not found: {image_dir}")
        
        results = {}
        save_dir = Path('prediction_results') if visualize else None
        save_dir.mkdir(exist_ok=True) if visualize else None
        
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() in {fmt.value for fmt in ImageFormat}:
                try:
                    save_path = save_dir / f"{img_path.stem}_prediction.png" if visualize else None
                    result = self.predict(img_path, visualize, save_path)
                    results[img_path.name] = result
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {str(e)}")
                    results[img_path.name] = {'error': str(e)}
        
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: Dict, output_file: Path) -> None:
        """Save prediction results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {output_path}")
