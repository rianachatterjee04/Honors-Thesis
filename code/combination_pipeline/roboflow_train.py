import ultralytics
from ultralytics import YOLO
import yaml
import os
from pathlib import Path

def fine_tune_yolo_with_training_only(demovideo, base_model_path, train_folder='train', epochs=25):
    """Fine-tune YOLOv8 model using only training data (YOLO will handle internal validation split)"""
    
    # Define folder paths
    train_images = Path(demovideo) / train_folder / 'images'
    train_labels = Path(demovideo) / train_folder / 'labels'
    
    # Check if training folders exist
    if not train_images.exists() or not train_labels.exists():
        raise ValueError(f"Training folder does not exist: {train_images} or {train_labels}")
    
    # Count files
    train_img_count = len(list(train_images.glob('*.jpg'))) + len(list(train_images.glob('*.png')))
    train_label_count = len(list(train_labels.glob('*.txt')))
    
    print(f"Training images: {train_img_count}")
    print(f"Training labels: {train_label_count}")
    
    # Create dataset YAML with train path for both train and val (YOLO will auto-split)
    dataset_yaml_content = f"""# Dataset configuration using only training data
train: {train_images.absolute()}
val: {train_images.absolute()}  # Same as train, YOLO will auto-split

# Number of classes
nc: 3

# Class names
names:
  0: 'side walk'
  1: 'road'  
  2: 'street sign'
"""
    
    yaml_path = 'walkable_dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    
    print(f"\nCreated YAML file:")
    print(dataset_yaml_content)
    
    # Load your base model (the original walkable_model.pt)
    model = YOLO(base_model_path)
    
    # Training parameters optimized for CPU and small dataset
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,    # Reduced for CPU
        save_period=5,
        device='cpu',
        project='yolo_training',
        name='walkable_v3',  # Changed name to v3 to avoid confusion
        exist_ok=True,
        patience=5,
        cos_lr=True,
        augment=True,
        val=True,
        fraction=1.0,  # Use all data
    )
    
    # Evaluate the model
    metrics = model.val()
    
    # Save the final model with a different name
    output_path = 'walkable_model1.pt'  # Different name to avoid overwriting
    model.save(output_path)
    
    # Print results
    print(f"\nTraining completed!")
    print(f"Final model saved as: {output_path}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")
    
    return output_path, results

# Main execution
if __name__ == "__main__":
    # Your demovideo folder is in the current directory
    demovideo = 'demovideo'
    
    # Use the original walkable_model.pt as your base
    base_model = 'YOLO_VLM_switch_off/walkable_model.pt'  # Path to the original model
    
    # Fine-tune the model using only your training data
    final_model, results = fine_tune_yolo_with_training_only(
        demovideo,
        base_model,
        train_folder='train',
        epochs=25
    )
    
    print(f"\nFinal model saved as: {final_model}")