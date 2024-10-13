import sounddevice as sd
import librosa
import numpy as np
import joblib
import os

# Constants
FS = 44100  # Sample rate
DURATION = 4  # Duration of recording in seconds
MODEL_PATH = "D:\IOT Home auto mation project\model.pkl"
CONFIDENCE_THRESHOLD = 0.7  # Adjust the threshold for identification confidence

# Load the trained model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Extract MFCC features from an audio signal
def extract_mfcc_from_audio(audio):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=FS, n_mfcc=40).T, axis=0)
    return mfccs

# Predict speaker from audio and get the probabilities
def predict_speaker(model, audio):
    mfcc_features = extract_mfcc_from_audio(audio).reshape(1, -1)  # Reshape for prediction
    probabilities = model.predict_proba(mfcc_features)  # Get probabilities for each class
    predicted_class = model.predict(mfcc_features)
    return predicted_class, probabilities

# Record audio in real-time
def record_audio():
    print("Recording your voice for identification...")
    rec = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    sd.wait()
    return rec.flatten()  # Flatten to 1D array

if __name__ == "__main__":
    # Load the trained model
    model = load_model(MODEL_PATH)

    while True:
        # Record audio
        audio_data = record_audio()

        # Predict the speaker and get probabilities
        predicted_speaker, probabilities = predict_speaker(model, audio_data)

        # Retrieve the unique speaker names from the model
        unique_speakers = model.classes_

        # Get the name of the predicted speaker
        predicted_name = unique_speakers[predicted_speaker[0]]
        predicted_probability = probabilities[0][predicted_speaker[0]]  # Probability of the predicted class

        print(f"Predicted speaker: {predicted_name}")
        print(f"Probability: {predicted_probability:.2f}")

        # Determine identification success based on the confidence threshold
        if predicted_probability >= CONFIDENCE_THRESHOLD:
            print("Identification successful! Output: 1")
            identification_output = 1
        else:
            print("Identification failed! Output: 0")
            identification_output = 0

        # Print the output
        print(f"Identification Output: {identification_output}")

        # Ask if the user wants to continue
        continue_prompt = input("Do you want to identify another voice? (yes/no): ").strip().lower()
        if continue_prompt != 'yes':
            break
