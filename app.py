from flask import Flask, render_template, request
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            message = request.form['message'].strip().lower()  # Preprocess input
            
            if not message:
                return render_template('result.html', prediction="Error: Empty message!")

            # Convert text to sequence
            seq = tokenizer.texts_to_sequences([message])
            padded = pad_sequences(seq, maxlen=100)

            # Predict spam or not spam
            prediction = model.predict(padded)[0][0]
            logging.info(f"Input: {message} | Prediction Score: {prediction}")

            # Adjusting threshold for better spam detection
            result = "Spam" if prediction > 0.4 else "Not Spam"

            return render_template('result.html', prediction=result)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return render_template('result.html', prediction="Error: Unable to process request!")

if __name__ == '__main__':
    app.run(debug=True)
