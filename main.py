import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import models
from imagenet import label_defs

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

st.title('Random image classifier')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    prepocess_func, model, model_name, resolution = models.load_random_model()
    print(f'Running {model_name}')
    img_arr = prepocess_func(np.array(img.resize((resolution, resolution))))
    img_arr = np.expand_dims(img_arr, 0)
    preds: np.ndarray = model.predict(img_arr)
    prob = preds.max()
    pred_class = preds.argmax()
    pred_class_name = label_defs[pred_class]
    st.text(f'Model: {model_name}\nPredicted class: {pred_class_name}\nProbability:'
            f' {prob * 100:.02f}%')
    st.image(img, width=400)
