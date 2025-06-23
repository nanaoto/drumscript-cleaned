# DrumScript/drum_classifier/model_trainer.py

import os
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import your modules
from drum_classifier.data_preparer import prepare_dataset
from drum_classifier.drum_model import DrumClassifier

def train_and_evaluate_model(data_dir: str, model_save_path: str, scaler_save_path: str,
                             label_map_save_path: str, sr: int = 22050,
                             segment_length_seconds: float = 0.2, model_type: str = 'random_forest'):
    """
    Trains and evaluates a drum classification model.

    Args:
        data_dir (str): Path to the directory containing drum sound subfolders.
        model_save_path (str): Path to save the trained model.
        scaler_save_path (str): Path to save the fitted StandardScaler.
        label_map_save_path (str): Path to save the label mapping.
        sr (int): Sample rate for audio processing.
        segment_length_seconds (float): Length of audio segment for feature extraction.
        model_type (str): Type of scikit-learn model to use ('random_forest', 'svm', 'mlp').
    """
    print("Starting model training and evaluation process...")

    # 1. Prepare Data
    X, y, scaler, label_map = prepare_dataset(data_dir, sr=sr, segment_length_seconds=segment_length_seconds)

    if X.shape[0] == 0:
        print("No data prepared. Exiting training process.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # 2. Initialize and Train Model
    classifier = DrumClassifier(model_type=model_type)
    classifier.train(X_train, y_train)

    # 3. Evaluate Model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation ({model_type}):")
    print(f"Accuracy: {accuracy:.4f}")

    # Reverse label map for human-readable report
    id_to_label = {v: k for k, v in label_map.items()}
    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 4. Save Model, Scaler, and Label Map
    print("\nSaving model assets...")
    models_dir = os.path.dirname(model_save_path)
    os.makedirs(models_dir, exist_ok=True)

    classifier.save_model(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to: {scaler_save_path}")
    with open(label_map_save_path, 'w') as f:
        json.dump(label_map, f)
    print(f"Label map saved to: {label_map_save_path}")

    print("\nModel training and evaluation finished.")


if __name__ == "__main__":
    # Determine project root dynamically
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

    # Define paths for data and saved models
    data_directory = os.path.join(project_root, "training_data")
    model_save_directory = os.path.join(project_root, "models")
    
    # Ensure 'models' directory exists
    os.makedirs(model_save_directory, exist_ok=True)

    model_file = os.path.join(model_save_directory, "drum_classifier_model.joblib")
    scaler_file = os.path.join(model_save_directory, "scaler.joblib")
    label_map_file = os.path.join(model_save_directory, "label_map.json")

    try:
        # You can change model_type here: 'random_forest', 'svm', 'mlp'
        train_and_evaluate_model(
            data_dir=data_directory,
            model_save_path=model_file,
            scaler_save_path=scaler_file,
            label_map_save_path=label_map_file,
            model_type='random_forest' # You can change this to 'svm' or 'mlp'
        )
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()