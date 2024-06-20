
import tensorflow as tf

import re
import string
import os

import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer 


model = TFSMLayer(
    os.path.join(os.path.dirname(__file__), '../../sentiment_model'), 
    call_endpoint='serving_default'
)


def preprocess_text(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )
