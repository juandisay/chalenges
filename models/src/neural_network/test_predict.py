import torch
import json

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from neural_network.model import DogClassifier
from neural_network.trainer import ModelTrainer
from neural_network.config import Config



class ModelPredictor:
    def __init__(self):
        self.config = Config()
        self.device = self.config.get_device()
        self.class_mapping = self._load_class_names()
        self.model = self._load_model()
        
    def _load_class_names(self):
        """Load class names from saved JSON file"""
        try:
            with open(self.config.CLASS_NAMES_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception("Class names file not found! Please train the model first.")
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Initialize model with correct number of classes
            num_classes = len(self.class_mapping['class_names'])
            model = DogClassifier(num_classes).to(self.device)
            
            # Load trained weights
            model.load_state_dict(torch.load(
                self.config.MODEL_SAVE_PATH, 
                map_location=self.device
            ))
            model.eval()
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict_image(self, image_path):
        """Predict single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = self._get_transform()
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top 5 predictions
                top_prob, top_class = torch.topk(probabilities, 5)
                top_prob = top_prob.squeeze().cpu().numpy()
                top_class = top_class.squeeze().cpu().numpy()
            
            # Convert indices to class names
            class_names = self.class_mapping['class_names']
            predictions = [(class_names[idx], prob) 
                         for idx, prob in zip(top_class, top_prob)]
            
            return predictions, image
            
        except Exception as e:
            raise Exception(f"Error predicting image: {str(e)}")
    
    def _get_transform(self):
        """Get validation transform"""
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def plot_prediction(image, predictions, save_path=None):
    """Plot image with predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    breeds, probs = zip(*predictions)
    y_pos = range(len(breeds))
    
    plt.barh(y_pos, probs)
    plt.yticks(y_pos, breeds)
    plt.xlabel('Probability')
    plt.title('Top 5 Predictions')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Initialize predictor
    print("Loading model and class names...")
    predictor = ModelPredictor()
    
    # Directory containing test images
    test_dir = Path("test_images")  # Change this to your test images directory
    
    # Create output directory for results
    output_dir = Path("loc_prediction_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process each image in test directory
    for img_path in test_dir.glob("*.jpg"):
        try:
            print(f"\nProcessing {img_path.name}...")
            
            # Make prediction
            predictions, image = predictor.predict_image(img_path)
            
            # Print results
            print("\nPrediction Results:")
            for i, (breed, prob) in enumerate(predictions, 1):
                print(f"{i}. {breed}: {prob:.2%}")
            
            # Plot and save results
            save_path = output_dir / f"{img_path.stem}_prediction.png"
            plot_prediction(image, predictions, save_path)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    main()