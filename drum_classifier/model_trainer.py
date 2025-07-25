# DrumScript/drum_classifier/model_trainer.py
# __backup4__sat12jul2025__cnn

import os
import numpy as np
import joblib
import json
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tqdm import tqdm # For progress bars during feature extraction


# --- Configuration ---
SAMPLE_RATE = 22050
SEGMENT_LENGTH_SECONDS = 0.2
# Define all UNIQUE drum types that we expect to classify.
# This must match the ALL_DRUM_TYPES list used in process_enst_dataset.py
ALL_DRUM_TYPES = sorted(['kick', 'snare', 'hi-hat', 'crash', 'ride', 'tom']) # Ensure this matches process_enst_dataset.py


# --- Feature Extraction Helper ---
def _extract_features(audio_path, sr, segment_length_seconds):
    """
    Extracts MFCC features from an audio segment.
    """
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        # Ensure the segment has the expected length, pad if necessary
        if len(audio) < int(segment_length_seconds * sr):
            padding = int(segment_length_seconds * sr) - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > int(segment_length_seconds * sr):
            audio = audio[:int(segment_length_seconds * sr)]

        # Extract MFCCs
        # n_mfcc=20 is a common choice for speech/audio classification
        # hop_length and n_fft impact the number of frames; keep consistent
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
        
        # Add delta (first derivative) MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)

        # Other features (often useful for drum classification)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=2048, hop_length=512)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=2048, hop_length=512)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512) # Root Mean Square energy

        # Concatenate all features
        # Flatten each feature to a 1D array before concatenating
        # This is suitable for a fully connected layer or for traditional ML models
        # For CNNs, you'd stack them as 2D arrays (e.g., [n_features, n_frames])
        
        # For CNN input, we want features over time.
        # We need to decide on a fixed number of frames per segment.
        # Given SEGMENT_LENGTH_SECONDS=0.2 and sr=22050, hop_length=512
        # num_frames = ceil(0.2 * 22050 / 512) = ceil(4410 / 512) = 9
        # Ensure all features have consistent number of frames.
        
        # Pad or truncate features to ensure consistent shape (e.g., 9 frames)
        target_frames = int(np.ceil(segment_length_seconds * sr / 512)) # 9 frames for 0.2s, 22050 sr, 512 hop
        
        def _pad_or_truncate(feature_array, target_frames_val):
            if feature_array.shape[1] < target_frames_val:
                padding = target_frames_val - feature_array.shape[1]
                return np.pad(feature_array, ((0, 0), (0, padding)), mode='constant')
            elif feature_array.shape[1] > target_frames_val:
                return feature_array[:, :target_frames_val]
            return feature_array

        mfccs = _pad_or_truncate(mfccs, target_frames)
        delta_mfccs = _pad_or_truncate(delta_mfccs, target_frames)
        chroma = _pad_or_truncate(chroma, target_frames)
        spectral_centroid = _pad_or_truncate(spectral_centroid, target_frames)
        spectral_bandwidth = _pad_or_truncate(spectral_bandwidth, target_frames)
        rolloff = _pad_or_truncate(rolloff, target_frames)
        zero_crossing_rate = _pad_or_truncate(zero_crossing_rate, target_frames)
        rms = _pad_or_truncate(rms, target_frames)

        # Stack features for CNN input: (n_features, n_frames) -> (n_frames, n_features) if time is primary dim
        # Or (n_features, n_frames, 1) if using Conv1D where last dim is channel
        
        # Concatenate along the feature dimension (axis=0)
        # Resulting shape: (total_features, n_frames)
        combined_features = np.vstack([
            mfccs,
            delta_mfccs,
            chroma,
            spectral_centroid,
            spectral_bandwidth,
            rolloff,
            zero_crossing_rate,
            rms
        ])
        
        # Reshape for Conv1D: (n_frames, n_features_per_frame, 1_channel)
        # Transpose to (n_frames, n_features) and then add channel dimension
        return combined_features.T[:, :, np.newaxis] # Shape will be (9, X, 1)
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        # Return a zero-filled array of the expected shape if extraction fails
        # n_mfcc (20) + delta_mfcc (20) + chroma (12) + centroid (1) + bandwidth (1) + rolloff (1) + zcr (1) + rms (1) = 57 features
        target_frames = int(np.ceil(segment_length_seconds * sr / 512))
        return np.zeros((target_frames, 57, 1)) # Default feature shape


# --- Model Definition (CNN) ---
def build_cnn_model(input_shape, num_labels):
    """
    Builds a Convolutional Neural Network (CNN) model for multi-label classification.
    Input shape: (n_frames, n_features, 1)
    """
    model = Sequential([
        # First Conv1D layer
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),

        # Second Conv1D layer
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),

        Flatten(), # Flatten the output of the convolutional layers

        Dense(256, activation='relu'),
        Dropout(0.5),

        # Output layer for multi-label classification
        Dense(num_labels, activation='sigmoid') # Sigmoid for multi-label
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(), # Use BinaryCrossentropy for multi-label
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    return model


# --- Training and Evaluation ---
def train_and_evaluate_model(data_dir: str, model_save_path: str, scaler_save_path: str, label_map_save_path: str):
    """
    Trains and evaluates a multi-label drum classification model.
    """
    print("Starting multi-label model training and evaluation process...")

    # Path to the processed multi-label CSV and audio segments
    processed_enst_dir = os.path.join(data_dir, "ENST_processed")
    multi_label_csv_path = os.path.join(processed_enst_dir, "multi_label_events.csv")
    
    if not os.path.exists(multi_label_csv_path):
        raise FileNotFoundError(f"Multi-label metadata CSV not found: {multi_label_csv_path}")

    # Load metadata
    print(f"Loading multi-label metadata from: {multi_label_csv_path}")
    df = pd.read_csv(multi_label_csv_path)

    # Prepare features and labels
    X = []
    y_labels = [] # To store list of labels per event
    
    print(f"Extracting features from {len(df)} audio segments...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Feature Extraction"):
        audio_segment_path = os.path.join(processed_enst_dir, row['audio_segment_filename'])
        features = _extract_features(audio_segment_path, SAMPLE_RATE, SEGMENT_LENGTH_SECONDS)
        X.append(features)
        
        # Parse multi-label string back to list
        labels_str = row['labels'].strip("[]").replace("'", "").split(', ')
        y_labels.append(labels_str)

    X = np.array(X)
    
    # Create a consistent label mapping
    # This ensures that even if a drum type isn't present in a batch, its column exists.
    label_map = {drum_type: i for i, drum_type in enumerate(ALL_DRUM_TYPES)}
    
    # Convert multi-labels to binary matrix format
    # Initialize y as a zero matrix of shape (num_samples, num_unique_drum_types)
    y = np.zeros((len(y_labels), len(ALL_DRUM_TYPES)), dtype=int)
    for i, labels in enumerate(y_labels):
        for label in labels:
            if label in label_map:
                y[i, label_map[label]] = 1 # Set 1 for present labels

    # Save the final label map for inference
    with open(label_map_save_path, 'w') as f:
        json.dump(label_map, f)
    print(f"Label map saved to: {label_map_save_path}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape X for StandardScaler (needs 2D array: samples, features)
    # The current X is (n_samples, n_frames, n_features_per_frame, 1)
    # For scaler, we need to flatten the last two dimensions to (n_samples, n_frames * n_features_per_frame)
    
    # Store original shape for reshaping back after scaling
    original_shape_train = X_train.shape
    original_shape_test = X_test.shape

    X_train_reshaped_for_scaler = X_train.reshape(original_shape_train[0], -1)
    X_test_reshaped_for_scaler = X_test.reshape(original_shape_test[0], -1)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped_for_scaler)
    X_test_scaled = scaler.transform(X_test_reshaped_for_scaler)

    # Reshape back to original CNN input shape
    X_train_scaled = X_train_scaled.reshape(original_shape_train)
    X_test_scaled = X_test_scaled.reshape(original_shape_test)

    # Save the scaler
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to: {scaler_save_path}")

    # Build and train CNN model
    input_shape = X_train_scaled.shape[1:] # (n_frames, n_features_per_frame, 1)
    num_labels = len(ALL_DRUM_TYPES)
    model = build_cnn_model(input_shape, num_labels)
    model.summary()

    print("\nTraining the CNN model...")
    history = model.fit(X_train_scaled, y_train,
                        epochs=50, # You can adjust the number of epochs
                        batch_size=32,
                        validation_split=0.1, # Use a portion of training data for validation
                        verbose=1)

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Evaluate the model
    print("\nEvaluating the model on the test set...")
    loss, accuracy, precision, recall, auc = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\n--- Model Evaluation Results ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Binary Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Classification report for more detailed metrics per label
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs > 0.5).astype(int) # Convert probabilities to binary predictions

    # Invert label_map to get drum type names from indices for the report
    idx_to_label = {i: drum_type for drum_type, i in label_map.items()}
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())] # Ensure order matches y columns

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print("\nMulti-label model training and evaluation finished.")
    print("\n---------------------------------------------------")


if __name__ == "__main__":
    # Determine project root dynamically
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from drum_classifier/ to DrumScript/
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

    # Define paths for data and saved models
    data_directory = os.path.join(project_root, "training_data") # Points to 'training_data' which contains 'ENST_processed'
    model_save_directory = os.path.join(project_root, "models")
    
    # Ensure 'models' directory exists
    os.makedirs(model_save_directory, exist_ok=True)

    # File paths for saving
    model_file = os.path.join(model_save_directory, "multi_label_drum_classifier_model.h5") # Changed extension for Keras model
    scaler_file = os.path.join(model_save_directory, "multi_label_scaler.joblib")
    label_map_file = os.path.join(model_save_directory, "multi_label_label_map.json")

    try:
        train_and_evaluate_model(
            data_dir=data_directory,
            model_save_path=model_file,
            scaler_save_path=scaler_file,
            label_map_save_path=label_map_file
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run 'process_enst_dataset.py' first to prepare the data.")
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging