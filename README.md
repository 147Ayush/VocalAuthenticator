
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

## Applications
- **Voice-Activated Security**: Can be used in systems requiring voice authentication, such as smart home systems or bank accounts.
- **Voice Assistants**: Enhances voice-activated assistants like Google Assistant, Alexa, or Siri by adding user-specific authentication.
- **Access Control**: Suitable for environments like offices or restricted areas where voice can act as a biometric key for entry.
- **Telecommunications**: Can be integrated into call centers for automatic speaker recognition and authentication.
- **Healthcare**: Secure patient data access by healthcare professionals using voice verification.

## Challenges Addressed
    - **Noise Robustness**:
        The model incorporates noise in the training phase to handle different levels of background interference, such as office sounds, street noise, or chatter,         ensuring accuracy even in suboptimal conditions.
        
    - **Scalability**:
        The system can be scaled to support hundreds or even thousands of speakers with minimal degradation in performance, making it suitable for large 
        organizations.
        
    - **Real-Time Performance**:
        The model is designed to provide near real-time predictions, making it suitable for live environments where quick decisions are critical.

## Future Improvements
While the current system is highly effective in voice recognition and authentication, there are several avenues for future improvement:

- **Integration with Other Biometric Systems**: Combine with facial recognition or fingerprint scanning for multi-factor authentication.
- **Deep Learning Model Optimization**: Further refine the CNN or experiment with other architectures such as RNNs (Recurrent Neural Networks) for sequential data     processing.
- **Language Independence**: Expand the system’s capabilities to recognize voices across multiple languages and accents.

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
