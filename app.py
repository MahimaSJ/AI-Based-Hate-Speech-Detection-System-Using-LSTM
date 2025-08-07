import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import io
import matplotlib.pyplot as plt
import dill


model = tf.keras.models.load_model('lstm_hate_speech_model.h5',compile=False)

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = dill.load(handle)


labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'No Hate/Offense'}

def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)  
    pred = model.predict(padded)
    predicted_class = np.argmax(pred, axis=1)[0]
    return labels[predicted_class], pred


st.set_page_config(page_title="Hate Speech Detector", page_icon="🚀")

st.title(" Hate Speech Detection App")

menu = ["Predict Text", "Upload File for Batch Prediction"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "Predict Text":
    user_input = st.text_area("Enter Text for Prediction:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to predict.")
        else:
            label, probs = predict_text(user_input)
            st.success(f"Prediction: **{label}**")

            st.subheader("Class Probabilities:")
            st.write({
                'Hate Speech': float(probs[0][0]),
                'Offensive Language': float(probs[0][1]),
                'No Hate/Offense': float(probs[0][2])
            })

            st.bar_chart({
                'Hate Speech': probs[0][0],
                'Offensive Language': probs[0][1],
                'No Hate/Offense': probs[0][2]
            })

elif choice == "Upload File for Batch Prediction":
    uploaded_file = st.file_uploader("Upload a TXT file (each line = one text)", type=['txt'])

    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        texts = content.splitlines()

        texts = [t for t in texts if t.strip() != ""]

        if not texts:
            st.error("Uploaded file is empty or invalid.")
        else:
            sequences = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=100)

            preds = model.predict(padded)
            pred_labels = np.argmax(preds, axis=1)
            pred_classes = [labels[label] for label in pred_labels]

            results_df = pd.DataFrame({'Text': texts, 'Prediction': pred_classes})
            st.dataframe(results_df)

            label_counts = results_df['Prediction'].value_counts()
            st.subheader("Prediction Distribution:")

            fig, ax = plt.subplots()
            ax.bar(label_counts.index, label_counts.values, color=['orange', 'green', 'red'])
            ax.set_ylabel('Count')
            ax.set_xlabel('Predicted Class')
            ax.set_title('Distribution of Predictions')
            st.pyplot(fig)

          
            csv = results_df.to_csv(index=False)
            st.download_button(label="Download Predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')
