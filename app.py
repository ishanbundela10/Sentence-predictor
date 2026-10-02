import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""


def generate_sentence(seed_text, word_count):
    sentence = seed_text.strip()

    for _ in range(word_count):
        next_word = predict_next_word(sentence)
        if not next_word:
            break
        sentence = f"{sentence} {next_word}"

    return sentence

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Sentence Predictor", layout="centered")

st.title("🧠 Sentence Predictor (LSTM)")
st.write("Enter a starting phrase and the model will continue it into a sentence.")

user_input = st.text_area(
    "✍️ Starting phrase:",
    placeholder="Type a phrase here...",
    height=100,
)
word_count = st.slider("Words to generate", min_value=1, max_value=30, value=10)

if st.button("Generate Sentence"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentence = generate_sentence(user_input, word_count)
        st.success(sentence)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based sentence prediction using Streamlit")

# next word predictor:

# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Load saved files
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     model = load_model("lstm_model.h5")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     with open("max_len.pkl", "rb") as f:
#         max_len = pickle.load(f)
#     return model, tokenizer, max_len

# model, tokenizer, max_len = load_resources()

# # ------------------------------
# # Prediction function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

#     preds = model.predict(sequence, verbose=0)
#     predicted_index = np.argmax(preds)

#     for word, index in tokenizer.word_index.items():
#         if index == predicted_index:
#             return word
#     return ""

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.set_page_config(page_title="Next Word Prediction", layout="centered")

# st.title("🧠 Next Word Prediction (LSTM)")
# st.write("Enter a sentence and the model will predict the **next word**.")

# user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

# if st.button("Predict Next Word"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         next_word = predict_next_word(user_input)
#         st.success(f"**Predicted Next Word:** {next_word}")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.caption("LSTM-based Next Word Prediction using Streamlit")