import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

voc_size = 5000
max_sent_length = 20

@st.cache(allow_output_mutation=True)
def load_model(model_type):
    if model_type == 'Model 1':
        model = tf.keras.models.load_model('cnnmodel.hdf5')
    else:
        model = tf.keras.models.load_model('Model2.weights.hdf5')
    return model

st.write("""# Sarcasm Detection System""")
sentence = st.text_input('Please enter a sentence.')

model_type = st.selectbox('Select Model Type', ('Model 1', 'Model 2'))

if sentence is None:
    st.text("Please enter a sentence.")
else:
    st.write('Inputted Sentence: ', sentence)
    predcorpus = [sentence]
    onehot_ = [one_hot(words, voc_size) for words in predcorpus] 
    embedded_docs = pad_sequences(onehot_, padding='pre', maxlen=max_sent_length)
    
    model = load_model(model_type)
    
    # Get labels based on probability 1 if p >= 0.01 else 0
    prediction = model.predict(embedded_docs)
    pred_labels = []
    if prediction >= 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)
    if pred_labels[0] == 1:
        output = 'Sarcasm Detected'
    else:
        output = 'No Sarcasm Detected'
    prediction = (str(prediction[0][0] * 100)) + '%'
    st.write("Prediction Accuracy: ", prediction)
    string = "OUTPUT: " + output
    st.success(string)
