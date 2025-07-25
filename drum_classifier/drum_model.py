import joblib # For saving/loading scikit-learn models
import numpy as np
import tensorflow as tf # For loading Keras model
import json # For loading label map
import librosa # Needed for feature processing within the classify_events context
from typing import List, Dict, Any

# Configuration constants (ideally these would be in a shared config file)
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2
N_FFT = 1024
HOP_LENGTH = 512
# Calculate MAX_FRAMES based on segment length and hop length, similar to model_trainer
# Adding 1 to account for potential partial frames or librosa's frame calculation specifics
MAX_FRAMES = int(np.ceil(SEGMENT_LENGTH_SECONDS * SAMPLE_RATE / HOP_LENGTH)) + 1
# TOTAL_FEATURE_DIM is the sum of dimensions of all extracted features per frame
# MFCCs (20) + Delta MFCCs (20) + Spectral Centroid (1) + Rolloff (1) + Chroma (12) + Tonnetz (6) + ZCR (1) = 61
TOTAL_FEATURE_DIM = 61


class DrumClassifier:
    """
    A class to encapsulate the drum classification model.
    Handles loading the pre-trained Keras model, scaler, and label map,
    and provides a method for classifying drum events.
    """
    def __init__(self, model_path: str = None, scaler_path: str = None, label_map_path: str = None):
        """
        Initializes the drum classifier model and loads components if paths are provided.

        Args:
            model_path (str, optional): Path to the pre-trained Keras model (.h5).
            scaler_path (str, optional): Path to the pre-fitted StandardScaler (.joblib).
            label_map_path (str, optional): Path to the label mapping JSON file.
        """
        self.model = None
        self.scaler = None
        self.label_map = None
        self.idx_to_label = None # To easily convert numerical indices back to labels
        self.drum_types = [] # List of drum type names, sorted by their index

        if model_path and scaler_path and label_map_path:
            self.load_model_components(model_path, scaler_path, label_map_path)

    def load_model_components(self, model_path: str, scaler_path: str, label_map_path: str):
        """
        Loads the Keras model, StandardScaler, and label map from specified paths.
        Raises IOError if any component fails to load.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded Keras model from: {model_path}")
        except Exception as e:
            raise IOError(f"Error loading Keras model from {model_path}: {e}")

        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from: {scaler_path}")
        except Exception as e:
            raise IOError(f"Error loading scaler from {scaler_path}: {e}")

        try:
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            # Create a reverse mapping for easy index-to-label conversion
            self.idx_to_label = {v: k for k, v in self.label_map.items()}
            # Get drum types sorted by their numerical index
            self.drum_types = sorted(list(self.label_map.keys()), key=lambda x: self.label_map[x])
            print(f"Loaded label map from: {label_map_path}")
            print(f"Loaded drum types (in order): {self.drum_types}")
        except Exception as e:
            raise IOError(f"Error loading label map from {label_map_path}: {e}")

    @staticmethod
    def _pad_or_truncate_features(features_dict: dict[str, np.ndarray]) -> np.ndarray:
        """
        Pads or truncates the feature time series from a single event to a fixed
        MAX_FRAMES length and stacks them for CNN input.

        Args:
            features_dict (dict[str, np.ndarray]): A dictionary of feature arrays
                                                   (e.g., mfccs, spectral_centroid).

        Returns:
            np.ndarray: The stacked and padded/truncated features,
                        transposed to (MAX_FRAMES, TOTAL_FEATURE_DIM) for Conv1D.
        """
        stacked_features_list = []
        # Ensure consistent order and dimensions for stacking as used during training
        feature_keys = ["mfccs", "delta_mfccs", "spectral_centroid",
                        "spectral_rolloff", "chroma", "tonnetz", "zero_crossing_rate"]

        for key in feature_keys:
            feature_array = features_dict.get(key)
            if feature_array is None or feature_array.size == 0:
                # If feature is missing or empty, create an empty array of the correct
                # feature dimension and then pad it.
                if key in ["mfccs", "delta_mfccs"]:
                    feature_dim = 20
                elif key == "chroma":
                    feature_dim = 12
                elif key == "tonnetz":
                    feature_dim = 6
                else: # spectral_centroid, spectral_rolloff, zero_crossing_rate
                    feature_dim = 1
                feature_array = np.empty((feature_dim, 0)) # Create empty array with correct feature_dim

            # Pad or truncate frames to MAX_FRAMES
            current_frames = feature_array.shape[1]
            if current_frames < MAX_FRAMES:
                # Pad with zeros at the end
                padding = MAX_FRAMES - current_frames
                padded_feature = np.pad(feature_array, ((0, 0), (0, padding)), mode='constant')
            else:
                # Truncate if longer than MAX_FRAMES
                padded_feature = feature_array[:, :MAX_FRAMES]

            stacked_features_list.append(padded_feature)

        # Vertically stack all features: result shape (TOTAL_FEATURE_DIM, MAX_FRAMES)
        stacked_features = np.vstack(stacked_features_list)
        # Transpose to (MAX_FRAMES, TOTAL_FEATURE_DIM) for Keras Conv1D layer input
        return stacked_features.T

    def classify_events(self, features_with_onsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classifies each detected drum event using the loaded model.

        Args:
            features_with_onsets (List[Dict[str, Any]]): A list of dictionaries,
                                                          each containing 'onset_time'
                                                          and 'features' (raw features dict).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing
                                  'onset_time' and 'drum_type' (classified label,
                                  which can be a list for multi-label classification).
        """
        if not all([self.model, self.scaler, self.label_map, self.idx_to_label]):
            raise RuntimeError("Model components (model, scaler, or label map) are not loaded. Call load_model_components first.")

        all_drum_events = []
        feature_vectors_for_prediction = []
        onset_times_for_prediction = []

        if not features_with_onsets:
            print("No features provided for classification.")
            return []

        for event in features_with_onsets:
            onset_time = event['onset_time']
            raw_features = event['features']

            # Preprocess features for the model: stack, pad/truncate
            processed_feature = self._pad_or_truncate_features(raw_features)

            # Flatten the (MAX_FRAMES, TOTAL_FEATURE_DIM) array into a 1D vector
            # for scaling with StandardScaler, which expects 2D (n_samples, n_features)
            flattened_feature = processed_feature.flatten()
            feature_vectors_for_prediction.append(flattened_feature)
            onset_times_for_prediction.append(onset_time)

        # Convert list of flattened features to a numpy array
        X_raw = np.array(feature_vectors_for_prediction)

        # Scale the features using the loaded scaler
        X_scaled = self.scaler.transform(X_raw)

        # Reshape for CNN input: (num_samples, MAX_FRAMES, TOTAL_FEATURE_DIM)
        # -1 infers the number of samples automatically
        X_final = X_scaled.reshape(-1, MAX_FRAMES, TOTAL_FEATURE_DIM)

        print(f"  Shape of input to model: {X_final.shape}")

        # Predict probabilities using the Keras model
        predictions_proba = self.model.predict(X_final)

        # Determine the drum type for each prediction (multi-label classification)
        # A common approach is to use a threshold (e.g., 0.5) on the sigmoid output
        threshold = 0.5
        predicted_labels_indices = (predictions_proba > threshold).astype(int)

        for i, pred_indices in enumerate(predicted_labels_indices):
            predicted_drum_types = []
            # Map predicted indices (from the model's output layer) back to drum type names
            for idx, is_present in enumerate(pred_indices):
                if is_present:
                    drum_type = self.idx_to_label.get(idx)
                    if drum_type:
                        predicted_drum_types.append(drum_type)

            # Fallback: If no drum type is predicted above the threshold,
            # take the class with the single highest probability.
            if not predicted_drum_types:
                highest_prob_idx = np.argmax(predictions_proba[i])
                highest_prob_drum_type = self.idx_to_label.get(highest_prob_idx)
                if highest_prob_drum_type:
                    predicted_drum_types.append(highest_prob_drum_type)
                else:
                    predicted_drum_types.append("unknown") # Final safety fallback

            all_drum_events.append({
                "onset_time": onset_times_for_prediction[i],
                "drum_type": predicted_drum_types # This is now a list of strings (e.g., ['kick', 'hi-hat'])
            })
            print(f"  Onset at {onset_times_for_prediction[i]:.2f}s classified as: {', '.join(predicted_drum_types)}")

        return all_drum_events