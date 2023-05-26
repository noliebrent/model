import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

voc_size = 5000
max_sent_length = 20

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('SDSFinalweights.best.hdf5')
    return model

model = load_model()
st.write("""# Sarcasm Detection System""")
sentence_type = st.radio("Select sentence type:", ("Type 1", "Type 2"))
sentence = st.text_input('Please enter a sentence.')

if sentence is not None and sentence != "":
    st.write('Inputted Sentence: ', sentence)
    predcorpus = [sentence]
    onehot_ = [one_hot(words, voc_size) for words in predcorpus]
    embedded_docs = pad_sequences(onehot_, padding='pre', maxlen=max_sent_length)
    
    # Get labels based on probability
    predictions = model.predict(embedded_docs)
    pred_labels = []
    for prediction in predictions:
        if prediction >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    if sentence_type == "Type 1":
        if pred_labels[0] == 1:
            output = 'Sarcasm Detected'
        else:
            output = 'No Sarcasm Detected'
        prediction = str(predictions[0][0] * 100) + '%'
        
    elif sentence_type == "Type 2":
        if pred_labels[0] == 1:
            output = 'Positive Sentiment Detected'
        else:
            output = 'Negative Sentiment Detected'
        prediction = str(predictions[0][0] * 100) + '%'
    
    st.write("Prediction Accuracy: ", prediction)
    string = "OUTPUT: " + output
    st.success(string)
else:
    st.text("Please enter a sentence.")
