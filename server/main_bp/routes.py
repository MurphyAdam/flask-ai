from datetime import datetime
from server.main_bp import main_bp
from flask import jsonify, render_template, request

import tensorflow as tf
from server.helpers.preprocessing import preprocess_text, model


@main_bp.route('/api')
def api():
    data = {
        'message': 'Hello world!',
        'timestamp': datetime.utcnow()
    }
    return jsonify(data), 200


@main_bp.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json()
    if 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    cleaned_text = preprocess_text(tf.convert_to_tensor(text)).numpy()
    cleaned_text_string = cleaned_text.decode("utf-8").strip()

    if cleaned_text_string == "":
        return jsonify({'error': 'Empty text after cleaning'}), 400

    # Convert the input text to a tensor with dtype=tf.string
    text_tensor = tf.convert_to_tensor([cleaned_text], dtype=tf.string)
    # Use the loaded model layer for inference
    prediction = model(text_tensor)
    # Extract the prediction value from the dictionary
    prediction_value = prediction['activation_4'].numpy()[0][0]
    # Binary classification with a threshold of 0.7
    sentiment = 'positive' if prediction_value >= 0.7 else 'negative'
    return jsonify({'text': text, 'sentiment': sentiment, 'weight': float(prediction_value)})


@main_bp.route('/')
def index():
    return render_template('index.html')
