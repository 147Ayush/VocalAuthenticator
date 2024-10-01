# Vocal Authentication System

This project implements a **Vocal Authentication System** that identifies speakers based on their voice samples. The system is trained using audio data, with additional noise added to simulate real-world environments.

## Features

- **Data Preprocessing**: Loads audio files and noise data from directories for preprocessing.
- **Data Visualization**: Visualizes the dataset distribution (audio files and noise) using bar charts.
- **Noise Addition**: Adds noise to audio data to improve model robustness in real-world conditions.
- **Model Architecture**: Uses a Convolutional Neural Network (CNN) to classify audio inputs.
- **Speaker Prediction**: Predicts the speaker of a given audio file and evaluates the model's accuracy.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/username/vocal-authenticator.git
    cd vocal-authenticator
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Audio Data**: Organize your audio files and noise files in the respective directories.
   
2. **Run the Model**:

    ```bash
    python model.py
    ```

3. **Visualization**: View data distribution visualizations within the notebook.

4. **Predict Speaker**:

    Call the `predict()` function, passing the path to an audio file and expected labels:

    ```python
    path = ['/path/to/audio.wav']
    labels = ['speaker_name']
    predict(path, labels)
    ```

## Results

The system will predict whether the speaker is correctly identified. It will print messages such as:

- **Welcome** if the speaker is correctly identified.
- **Sorry** if the speaker is misidentified.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
