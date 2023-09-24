

import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img)

        img = img.resize((28, 28))
        img = img.convert("L")                          # Преобразуем изображение в оттенки серого
        img_arr = image.img_to_array(img)               # Переводим картинку в массив
        zeros = np.count_nonzero(img_arr == 0)          # считаем чёрные пиксели
        if zeros < 300:                                 # если чёрных мало - значит чёрным по белому, 
            img_arr = 255 - img_arr                     # и поэтому инвертируем Ч-Б, т.к. модель
                                                        # натренирована на белым по чёрному

        x = np.expand_dims(img_arr, axis=0)             # добавляем измерение впереди
        #st.write(x.shape)
        return x
    else:
        return None

######################

st.title('Классификации изображений в облаке Streamlit')
model = load_model('model_fmr_all.h5')
x = load_image()

result = st.button('Распознать изображение')
if result:
    preds = model.predict(x)
    pred = np.argmax(preds)
    #st.write(preds)
    st.write(f'Распознана цифра: {pred}')

######################

