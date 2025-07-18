import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load large spam dataset
df = pd.read_csv("spam.csv", encoding="latin-1")  # Load dataset
df = df[['v1', 'v2']]  # Keep only relevant columns
df.columns = ['label', 'message']  # Rename columns

# Convert labels: 'spam' → 1, 'ham' → 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Extract messages and labels
texts = df['message'].values
labels = df['label'].values

# Tokenize text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=100)  # Padding sequences

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert labels to numpy array
y = np.array(labels)

# Define LSTM Model for Spam Detection
model = Sequential([
    Embedding(5000, 64, input_length=100),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(1, activation="sigmoid")  # Binary classification (Spam or Not Spam)
])

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model on large dataset
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Save Model
model.save("model.h5")
print("Spam Detection Model trained on large dataset and saved as model.h5")
