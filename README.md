
# Vocal Authentication System

This project implements a **Vocal Authentication System** that identifies and authenticates speakers based on their voice samples. The system is trained on clean audio data, with additional noise incorporated to simulate real-world conditions, making it robust in diverse acoustic environments.

## Features

- **Data Preprocessing**:
    - Loads and organizes both voice and noise audio files from directories.
    - Converts audio data into a format suitable for neural network training (e.g., Mel spectrograms or FFT).
    - Splits the dataset into training and validation subsets to evaluate model performance.

- **Data Visualization**:
    - Visualizes the distribution of the audio data using bar charts, showcasing the number of files per class (voice and noise) using Plotly.
    - Ensures a balanced dataset, which helps to prevent overfitting or underfitting during training.

- **Noise Addition**:
    - Adds random noise to the audio data during training to increase the model’s robustness.
    - Simulates real-world conditions by blending noise files with clean speech data.
    - Helps the model generalize better to unseen data, especially in noisy environments.

- **Model Architecture**:
    - Utilizes a Convolutional Neural Network (CNN) designed for audio data classification.
    - The model is optimized for recognizing unique vocal features across different speakers.
    - Features layers like 1D convolutions, batch normalization, and dense layers to classify speaker inputs.

- **Speaker Prediction**:
    - Allows you to input a new voice sample and predict the speaker.
    - Evaluates model accuracy by comparing predicted speakers to actual labels.
    - Provides detailed output including whether the speaker was correctly identified, and logs cases where the prediction was incorrect.

## Installation

1. **Clone the repository**:
   
    Clone this project to your local machine using the following command:

    ```bash
    git clone https://github.com/username/vocal-authenticator.git
    cd vocal-authenticator
    ```

2. **Install dependencies**:

    Use the following command to install all required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Audio Data**:
   
    Organize your audio files and noise files in their respective directories:
    - `/audio` for clean voice files.
    - `/noise` for noise files to be used for augmentation.

2. **Run the Model**:

    To run the vocal authenticator, use:

    ```bash
    python model.py
    ```

3. **Visualize Data**:
   
    The model will visualize the dataset's distribution using bar charts, ensuring the data is balanced.

4. **Predict Speaker**:

    To predict a speaker, use the `predict()` function, passing the path to an audio file and the expected labels:

    ```python
    path = ['/path/to/audio.wav']
    labels = ['speaker_name']
    predict(path, labels)
    ```

## Results

The model will display results indicating whether the speaker was correctly identified:
- **Welcome** if the speaker was correctly recognized.
- **Sorry** if the speaker was not recognized.

## Dependencies

- `TensorFlow`
- `Keras`
- `NumPy`
- `Plotly`
- `Librosa`
- `Matplotlib`

## Project Structure

- `model.py`: Main script that contains the model definition, data processing, and prediction functions.
- `/audio`: Directory containing the clean audio files.
- `/noise`: Directory containing noise files to augment the dataset.

## DataSet

To train the vocal authenticator, you can get the dataset from Kaggle or use the following link to download it:

[Download the dataset from Kaggle](https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset)

Make sure to organize your audio files and noise files in their respective directories:

/audio: For clean voice files.
/noise: For noise files to augment the dataset.
