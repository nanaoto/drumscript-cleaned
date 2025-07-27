# DrumScript/drum_classifier/drum_model.py

import joblib # For saving/loading scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier # Multi-layer Perceptron
import numpy as np

class DrumClassifier:
    """
    A class to encapsulate the drum classification model.
    Supports different scikit-learn classifiers.
    """
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initializes the drum classifier model.

        Args:
            model_type (str): Type of scikit-learn model to use ('random_forest', 'svm', 'mlp').
            **kwargs: Additional arguments for the chosen scikit-learn classifier.
        """
        self.model = None
        self.model_type = model_type

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, **kwargs)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42, **kwargs)
        elif model_type == 'mlp':
            #self.model = MLPClassifier(random_state=42, max_iter=500, **kwargs)
            self.model = MLPClassifier(random_state=42, **kwargs) # Removed max_iter=500 here due to conflict with line 130: Python sees both max_iter=500 and max_iter=200 being passed to the MLPClassifier constructor simultaneously, which causes the TypeError.
            
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose from 'random_forest', 'svm', 'mlp'.")

        print(f"Initialised DrumClassifier with model type: {model_type}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the classification model.

        Args:
            X (np.ndarray): Feature matrix (samples x features).
            y (np.ndarray): Label vector (numerical labels).
        """
        if self.model is None:
            raise RuntimeError("Model not initialised. Call __init__ first.")
        print(f"Training {self.model_type} model...")
        self.model.fit(X, y)
        print("Model training complete.")

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new, unseen feature data.

        Args:
            X_new (np.ndarray): New feature matrix (samples x features).

        Returns:
            np.ndarray: Predicted labels for the input samples.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.model.predict(X_new)

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for new, unseen feature data.

        Args:
            X_new (np.ndarray): New feature matrix (samples x features).

        Returns:
            np.ndarray: Predicted class probabilities (samples x n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_new)
        else:
            raise NotImplementedError(f"Model type {self.model_type} does not support predict_proba.")

    def save_model(self, file_path: str):
        """
        Saves the trained model to a file.

        Args:
            file_path (str): Path to save the model.
        """
        joblib.dump(self.model, file_path)
        print(f"Model saved to: {file_path}")

    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads a trained model from a file.

        Args:
            file_path (str): Path to the saved model.

        Returns:
            DrumClassifier: An instance of DrumClassifier with the loaded model.
        """
        instance = cls() # Create a dummy instance to hold the loaded model
        instance.model = joblib.load(file_path)
        print(f"Model loaded from: {file_path}")
        return instance

# Example usage (for testing this module independently if needed)
if __name__ == "__main__":
    print("Running drum_model.py example...")
    # This is a minimal example, usually you'd train with a real dataset
    # For a full test, run model_trainer.py

    # Create dummy data
    X_dummy = np.random.rand(10, 50) # 10 samples, 50 features
    y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # Binary labels

    # Test RandomForestClassifier
    rf_classifier = DrumClassifier(model_type='random_forest')
    rf_classifier.train(X_dummy, y_dummy)
    predictions = rf_classifier.predict(X_dummy[:2])
    print(f"Random Forest predictions for first 2 samples: {predictions}")

    # Test SVM Classifier
    svm_classifier = DrumClassifier(model_type='svm')
    svm_classifier.train(X_dummy, y_dummy)
    predictions = svm_classifier.predict(X_dummy[:2])
    print(f"SVM predictions for first 2 samples: {predictions}")

    # Test MLP Classifier
    mlp_classifier = DrumClassifier(model_type='mlp', hidden_layer_sizes=(10,), max_iter=200)
    mlp_classifier.train(X_dummy, y_dummy)
    predictions = mlp_classifier.predict(X_dummy[:2])
    print(f"MLP predictions for first 2 samples: {predictions}")

    # Example of saving and loading
    try:
        import os
        models_dir = "temp_models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "test_model.joblib")
        rf_classifier.save_model(model_path)
        loaded_model = DrumClassifier.load_model(model_path)
        assert np.array_equal(loaded_model.predict(X_dummy[:1]), rf_classifier.predict(X_dummy[:1]))
        print("Model save/load successful!")
    except Exception as e:
        print(f"Error during save/load test: {e}")
    finally:
        if os.path.exists(models_dir):
            import shutil
            shutil.rmtree(models_dir)
            print(f"Cleaned up {models_dir}")

    print("drum_model.py example finished.")