
# **Speaker Identification System**

## **Overview**

This project implements a **Speaker Identification System** using **Multi-Layer Perceptron (MLP)** neural networks. The system takes voice samples from multiple speakers, extracts MFCC (Mel-frequency cepstral coefficients) features from the audio data, and trains a binary classifier to recognize a target speaker. The classifier is optimized for high accuracy.

## **Project Structure**

```
├── Data                    # Directory for storing audio data and processed CSVs
│   ├── complete_data.csv    # The dataset combining all speaker's data
├── model_mlp.h5             # Trained MLP model
├── Dataset_generator.py     # Script to generate and process audio data
├── model_trainer.py         # Script to train the MLP model
├── README.md                # This readme file
└── requirements.txt         # List of dependencies required to run the project
```

## **Features**

- Record real-time voice samples of speakers
- Extract **MFCC features** from audio data for speaker identification
- Train a **MLP neural network** to classify a specific target speaker using their voice
- Use **early stopping** to avoid overfitting and save the best-performing model
- Evaluate the model with **accuracy score** and **classification report**

## **Setup and Installation**

### Prerequisites

1. **Python 3.x** must be installed on your system.
2. Install the required dependencies using the provided `requirements.txt` file. You can do this by running:

    ```bash
    pip install -r requirements.txt
    ```

### Dependencies

- **Librosa**: For audio processing and MFCC extraction
- **Sounddevice**: For recording real-time audio
- **Keras** and **TensorFlow**: For building and training the MLP model
- **Scikit-learn**: For data processing, model evaluation, and splitting
- **Pandas & Numpy**: For data handling and numerical operations

### Setting Up the Project

1. Clone the repository or download the project files.
2. Ensure the folder structure is correct, with a `Data` directory to store audio and CSV files.
3. Update paths to the dataset and model in the scripts if needed.
4. Place your voice data or generate it using the provided `Dataset_generator.py` script.

## **Usage**

### Step 1: Generate Dataset

Run the `Dataset_generator.py` script to record audio data and extract MFCC features.

```bash
python Dataset_generator.py
```

- You'll be prompted to enter the speaker's name and the number of recordings.
- The script will save both the audio files and a CSV containing the MFCC features for each speaker.

### Step 2: Train the Model

Once the dataset is ready, run the `model_trainer.py` script to train the MLP neural network.

```bash
python model_trainer.py
```

- This script will load the `complete_data.csv` file, preprocess the data, and train the MLP neural network.
- The trained model will be saved as `model_mlp.h5`.

### Step 3: Model Evaluation

The accuracy and classification report will be printed to the console during training. These metrics evaluate how well the model identifies the target speaker.

## **Customization**

- **Target Speaker**: Modify the `target_speaker` variable in the `model_trainer.py` script to change the speaker you want the system to recognize.
- **Neural Network Architecture**: You can tweak the number of layers, units, and activation functions in the `train_mlp()` function to improve the model's performance.

## **Model Saving**

The trained MLP model is saved as `model_mlp.h5` using the Keras API. You can load this model later for prediction tasks or further training.

## **References**

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Keras Documentation](https://keras.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## **Contributing**

Feel free to contribute to this project by opening issues or submitting pull requests. Improvements, bug fixes, and suggestions are welcome!

---

## **Author**

- Ayush soni

---
