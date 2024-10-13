import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Constants
COMPLETE_CSV = "D:\\IOT Home auto mation project\\Data\\complete_data.csv"
MODEL_OUTPUT_PATH = "D:\\IOT Home auto mation project\\model.h5"

# Load the dataset
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Preprocess the dataset for multi-class classification
def preprocess_data(df):
    print("Unique values in the DataFrame before processing:")
    print(df['speaker'].unique())

    X = df.iloc[:, :-1].values  # Features (MFCCs)
    y = df['speaker'].astype(str).values  # Convert to string

    # Print unique values in 'y'
    print("Unique values in the speaker column:")
    print(np.unique(y))

    # Encode speaker labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder.classes_

# Reshape input for CNN
def reshape_input(X):
    # Reshape X to be (samples, time_steps, features, channels)
    # Here, we assume that MFCCs are structured as time steps (frames) and feature dimensions.
    num_samples = X.shape[0]
    time_steps = 20  # Modify according to your data
    num_features = X.shape[1] // time_steps  # Assuming equal splits for simplicity
    X_reshaped = X.reshape(num_samples, time_steps, num_features, 1)  # Add a channel dimension
    return X_reshaped

# Train the CNN model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape input for CNN
    X_train = reshape_input(X_train)
    X_test = reshape_input(X_test)

    # Define the CNN architecture
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(np.unique(y)), activation='softmax')  # Output layer for multi-class
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred_classes))
    print("Classification Report:\n", classification_report(y_test, y_pred_classes))

    return model

# Save the trained model
def save_model(model, output_path):
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    # Load data
    df = load_data(COMPLETE_CSV)

    # Preprocess data
    X, y_encoded, unique_labels = preprocess_data(df)

    # Train model
    model = train_model(X, y_encoded)

    # Save model
    save_model(model, MODEL_OUTPUT_PATH)

    # Print unique labels for reference
    print("Unique labels (speakers):", unique_labels)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# import joblib
# import os
#
# # Constants
# COMPLETE_CSV = "D:\\IOT Home auto mation project\\Data\\complete_data.csv"
# MODEL_OUTPUT_PATH = "D:\\IOT Home auto mation project\\model_mlp.h5"  # Save as .h5 for Keras
#
# # Load the dataset
# def load_data(csv_file):
#     df = pd.read_csv(csv_file)
#     return df
#
# # Preprocess the dataset for binary classification
# def preprocess_data(df, target_speaker):
#     print("Unique values in the DataFrame before processing:")
#     print(df['speaker'].unique())
#
#     X = df.iloc[:, :-1].values  # Features (MFCCs)
#     y = df['speaker'].astype(str).values  # Convert to string
#
#     print("Unique values in the speaker column after conversion to string:")
#     print(np.unique(y))
#
#     y_binary = np.array([1 if label == target_speaker else 0 for label in y])
#     return X, y_binary
#
# # Train the MLP model
# def train_mlp(X, y):
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Create the MLP model
#     model = Sequential()
#     model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
#     model.add(Dense(64, activation='relu'))  # Second hidden layer
#     model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
#
#     # Compile the model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     # Add early stopping to avoid overfitting
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#
#     # Train the model
#     history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16,
#                         callbacks=[early_stopping], verbose=1)
#
#     # Evaluate the model
#     y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Predict probabilities and convert to binary
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#
#     return model
#
# # Save the trained model
# def save_model(model, output_path):
#     model.save(output_path)
#     print(f"Model saved to {output_path}")
#
# if __name__ == "__main__":
#     os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
#
#     df = load_data(COMPLETE_CSV)
#     target_speaker = "Ayush"  # Change to your desired target speaker
#     X, y_binary = preprocess_data(df, target_speaker)
#
#     # Train model
#     model = train_mlp(X, y_binary)
#
#     # Save model
#     save_model(model, MODEL_OUTPUT_PATH)
