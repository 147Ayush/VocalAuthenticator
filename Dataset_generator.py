import sounddevice as sd
import librosa
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
import os

# Constants for recording
FS = 44100  # Sample rate
DURATION = 4  # Duration of recording in seconds
OUTPUT_DIR = "D:\IOT Home auto mation project\Data"
COMPLETE_CSV = os.path.join(OUTPUT_DIR, "complete_data.csv")


# Record audio 'n' times and save them as .wav files
def record_audio():
    speaker_name = input("Enter the speaker's name: ")
    print(f"Recording for speaker: {speaker_name}")
    n = int(input("How many recordings? "))

    for i in range(n):
        print(f"Recording {i + 1} of {n} started...")

        rec = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
        sd.wait()

        file_name = os.path.join(OUTPUT_DIR, f"{speaker_name}_{i}.wav")
        write(filename=file_name, rate=FS, data=rec)

        print(f"Recording {i + 1} saved as {file_name}")
        if i < n - 1:  # Ask for more recordings if it's not the last one
            choice = int(input("Record again? 1 for yes, 0 for no: "))
            if choice == 0:
                break


# Extract MFCCs and create a CSV file for the user's recordings
def create_mfcc_csv(speaker_name):
    df = pd.DataFrame(columns=range(40))  # 40 MFCC features
    for i in range(10):  # Modify range based on the number of recordings
        file_name = os.path.join(OUTPUT_DIR, f"{speaker_name}_{i}.wav")
        if os.path.exists(file_name):
            mfccs = extract_mfcc(file_name)
            df.loc[len(df)] = mfccs
        else:
            print(f"File not found: {file_name}")

    output_file = os.path.join(OUTPUT_DIR, f"{speaker_name}.csv")
    df.to_csv(output_file, index=False)
    print(f"MFCC data saved to {output_file}")


# Append the speaker's data to the complete dataset
def append_individual_to_complete_csv(speaker_name):
    individual_csv = os.path.join(OUTPUT_DIR, f"{speaker_name}.csv")

    if not os.path.exists(COMPLETE_CSV):
        print(f"{COMPLETE_CSV} not found, creating a new one...")
        pd.DataFrame(columns=range(40)).to_csv(COMPLETE_CSV, index=False)

    df_complete = pd.read_csv(COMPLETE_CSV)
    df_individual = pd.read_csv(individual_csv)

    # Add speaker name as a column
    df_individual['speaker'] = speaker_name

    df_combined = pd.concat([df_complete, df_individual], ignore_index=True)
    df_combined.to_csv(COMPLETE_CSV, index=False)
    print(f"Data from {individual_csv} appended to {COMPLETE_CSV}")


# Helper function to extract MFCCs from an audio file
def extract_mfcc(file, n_mfcc=40):
    audio, sr = librosa.load(file, sr=None)  # Load audio file
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfccs


# Example usage
while True:
    # record_audio()
    speaker_name = input("Enter the speaker's name for MFCC extraction: ")
    create_mfcc_csv(speaker_name)
    append_individual_to_complete_csv(speaker_name)

    # Check if the user wants to add another speaker
    another = input("Do you want to add another speaker? (yes/no): ").strip().lower()
    if another != 'yes':
        break
