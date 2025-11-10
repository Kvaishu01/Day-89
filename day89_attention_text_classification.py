# day89_attention_text_classification.py
# 🧠 Self-Attention Text Classification - Day 89

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Layer, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import streamlit as st

# -------------------------
# 🔹 Custom Self-Attention Layer
# -------------------------
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # Compute attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = a * x
        return output

# -------------------------
# 🔹 Sample Text Dataset
# -------------------------
texts = [
    "I love this movie", "This film was terrible", "Amazing acting and story",
    "Worst movie ever", "I really enjoyed it", "Boring and too long",
    "Fantastic direction", "Not good", "Wonderful soundtrack", "Awful plot"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Tokenize
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=6, padding="post")
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 🔹 Build Model
# -------------------------
input_layer = Input(shape=(X.shape[1],))
embedding = Embedding(input_dim=1000, output_dim=64)(input_layer)
attention = SelfAttention()(embedding)
x = GlobalAveragePooling1D()(attention)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# 🔹 Train Model
# -------------------------
st.write("🚀 Training Self-Attention Model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# -------------------------
# 🔹 Streamlit App
# -------------------------
st.title("🧠 Self-Attention Text Classifier")
st.write("Classify text as Positive or Negative using Attention Mechanism")

user_input = st.text_input("Enter text for classification:")
if user_input:
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=6, padding="post")
    pred = model.predict(padded)[0][0]
    sentiment = "😊 Positive" if pred > 0.5 else "😞 Negative"
    st.subheader(f"Prediction: {sentiment} ({pred:.2f})")
