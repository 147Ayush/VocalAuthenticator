import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from sklearn.preprocessing import LabelEncoder


COMPLETE_CSV = "D:\IOT Home auto mation project\Data\complete_data.csv"
MODEL_OUTPUT_PATH = "D:\IOT Home auto mation project\model.pkl"

# Load the dataset
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Preprocess the dataset for multi-class classification
def preprocess_data(df):
    # Check and print unique values to diagnose issues
    print("Unique values in the DataFrame before processing:")
    print(df['speaker'].unique())

    # Extract features and labels
    X = df.iloc[:, :-1].values  # Features (MFCCs)

    # Ensure all values in the 'speaker' column are strings
    y = df['speaker'].astype(str).values  # Convert to string

    # Debug information: Print unique values in 'y' after converting to string
    print("Unique values in the speaker column after conversion to string:")
    print(np.unique(y))

    # Encode speaker labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder.classes_  # Return the classes for future reference

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using Random Forest as an example classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model

# Save the trained model
def save_model(model, output_path):
    joblib.dump(model, output_path)
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
