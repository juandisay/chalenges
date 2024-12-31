from neural_network.model import DogClassifier
from neural_network.trainer import ModelTrainer
from neural_network.config import Config
from neural_network.dataset import DatasetManager

def main() -> ModelTrainer:
    # Initialize configuration
    config = Config()
    print(f"\nTraining Configuration:")
    print(f"Device: {config.get_device()}")
    
    # Setup data
    dataset_manager = DatasetManager(config)
    train_loader, val_loader, num_classes = dataset_manager.get_data_loaders()
    
    # Initialize model
    model = DogClassifier(num_classes).to(config.get_device())
    
    # Initialize trainer
    trainer = ModelTrainer(model, config)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Plot results
    trainer.plot_training_history(history)
    
    # Load and verify saved class names
    class_mapping = trainer.load_class_names()
    if class_mapping:
        print("\nLoaded class names:")
        print(f"Number of classes: {len(class_mapping['class_names'])}")
        print(f"First 5 classes: {class_mapping['class_names'][:5]}")
    
    return trainer

if __name__ == "__main__":
    main()
