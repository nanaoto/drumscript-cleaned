
# Drumscript/drum_classifier/drum_model.py
# 1D CNN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_drum_model(input_shape, num_classes):
    """
    Creates a Dense neural network model for drum classification.

    Args:
        input_shape (int): The number of features in the input vector.
        num_classes (int): The number of output drum classes.

    Returns:
        A compiled Keras model.
    """
    model = Sequential([
        # Input Layer
        tf.keras.Input(shape=(input_shape,)),

        # First Hidden Layer
        Dense(256, activation='relu'),
        BatchNormalization(), # Helps stabilise training
        Dropout(0.4), # Reduces overfitting

        # Second Hidden Layer
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        # Third Hidden Layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        # Output Layer
        # Sigmoid activation is used for multi-label classification
        Dense(num_classes, activation='sigmoid')
    ])

    # Compile the model
    # BinaryCrossentropy is the standard loss for multi-label problems
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'] # Can also use tf.keras.metrics.AUC()
    )

    return model
