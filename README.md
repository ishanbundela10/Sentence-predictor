# Sentence Predictor

A Streamlit app that uses a trained LSTM model to continue a user-provided phrase into a generated sentence.

## Overview

This project loads a saved Keras model and tokenizer from disk, then predicts the next word sequence based on an input phrase. The app is built with:

- Python
- TensorFlow / Keras
- NumPy
- Streamlit

## Project files

- `app.py` — Streamlit interface and prediction logic
- `lstm_model.h5` — trained LSTM model
- `tokenizer.pkl` — saved tokenizer used for text preprocessing
- `max_len.pkl` — saved sequence length used during model input preparation

## Requirements

Make sure you have Python 3.9+ installed.

Install the required packages:

```bash
pip install streamlit tensorflow numpy
```

## Run the app

From the project directory, start the app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

## Usage

1. Enter a starting phrase.
2. Choose how many words to generate.
3. Click `Generate Sentence`.
4. The model will continue the sentence based on the trained LSTM.

## Notes

- The model and tokenizer files are already included in the repository.
- The app uses cached model loading to reduce startup time.
- This project is intended for demo and learning purposes.
