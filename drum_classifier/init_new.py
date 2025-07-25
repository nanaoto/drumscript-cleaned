# DrumScript/drum_classifier/__init__.py
# (Add this to your existing __init__.py if it's not empty, or create it if it doesn't exist)

import numpy as np
import json
import joblib
import os
# You might need to import DrumClassifier if you plan to instantiate it here
# from .drum_model import DrumClassifier # Uncomment if you'll instantiate model within this function

def classify_events(detected_onsets_features: list[tuple[float, dict[str, np.ndarray]]],
                    model, # This will be your loaded model (RandomForest, SVM, MLP, or Keras CNN)
                    scaler, # This will be your loaded StandardScaler
                    label_map: dict[str, int]) -> list[dict[str, any]]:
    """
    Classifies detected drum onsets using a pre-trained model and returns
    a list of dictionaries, each containing 'onset_time' and 'drum_type'.

    Args:
        detected_onsets_features (list[tuple[float, dict[str, np.ndarray]]]):
            A list where each tuple contains (onset_time, features_dict).
            features_dict contains extracted features like 'mfccs', 'delta_mfccs'.
        model: The loaded trained drum classification model.
        scaler: The loaded StandardScaler used for feature scaling.
        label_map (dict[str, int]): A dictionary mapping drum type names to their
                                    numerical labels (e.g., {'kick': 0, 'snare': 1}).

    Returns:
        list[dict[str, any]]: A list of dictionaries, each with 'onset_time' (float)
                              and 'drum_type' (str).
    """
    classified_drum_events = []
    
    # Invert label_map for easy lookup of drum type by predicted label index
    # For multi-label, this might need adjustment if your model predicts a vector
    # For now, assuming single-label classification from scikit-learn models
    # This also assumes label_map maps string->int, we need int->string
    inverse_label_map = {v: k for k, v in label_map.items()}

    print(f"--- Detected Drum Events ---") # Keep this for consistent logging

    for onset_time, features_dict in detected_onsets_features:
        # Combine features into a single array for the scaler and model
        # Assuming features_dict contains 'mfccs' and 'delta_mfccs'
        # And assuming these are already flattened or are (n_features,) shape
        # For CNN, this part will need to ensure the correct input shape (samples, timesteps, features)
        
        # --- IMPORTANT: Feature flattening for scikit-learn models ---
        # If your features_dict contains 2D arrays (e.g., (n_mfcc, n_frames)),
        # you need to flatten them into a 1D array for scikit-learn models.
        # For a CNN, you'll need to reshape them differently.
        
        # For now, let's assume we need to flatten all features from features_dict
        # Adjust this logic based on the actual output shape of feature_extractor.py's extract_features
        
        # Example: Flattening MFCCs and delta_MFCCs (assuming they are 2D from feature_extractor)
        # Check actual shapes first. feature_extractor.py snippet shows 'mfccs' as (20, n_frames)
        
        # If your features are already in a single feature vector per onset from feature_extractor,
        # you can simplify this. Assuming they are dicts of np.ndarrays:
        
        # Concatenate all features into one vector. Ensure consistent order.
        # This part depends heavily on what 'feature_extractor.extract_features' returns.
        
        # Let's assume features_dict contains 'mfccs', 'delta_mfccs', etc.
        # And we need to flatten them into a single row vector for scaling and prediction.
        
        feature_vector = []
        for key in sorted(features_dict.keys()): # Sort keys for consistent order
            feature = features_dict[key]
            # Ensure feature is 1D or flatten it
            if feature.ndim > 1:
                feature_vector.extend(feature.flatten())
            else:
                feature_vector.extend(feature)
        
        # Convert to numpy array and reshape for single sample prediction
        single_sample_features = np.array(feature_vector).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(single_sample_features)

        # Predict the drum type
        # For scikit-learn models, predict returns a single label or array of labels
        prediction_index = model.predict(scaled_features)[0] # Get the first (and only) prediction

        # Map the prediction index back to a drum type string
        drum_type = inverse_label_map.get(prediction_index, "unknown") # Default to "unknown"

        print(f"  Onset at {onset_time:.2f}s classified as: {drum_type}")

        classified_drum_events.append({
            'onset_time': onset_time,
            'drum_type': drum_type
        })
    
    print(f"--- Inference Complete ---")
    return classified_drum_events