import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)

# Download and Prepare Dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# Display dataset directory structure
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# Read a sample positive review
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# Remove the unsupervised data directory
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Create Training, Validation, and Test Datasets
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size
)

# Display a few examples from the training dataset
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

# Custom Standardization Function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Text Vectorization Layer
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Prepare the Text-Only Dataset for Vectorization
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Vectorize the Text
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Display Vectorized Example
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# Display Vocabulary Information
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Vectorize the Datasets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Optimize Data Loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the Model
embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim, input_length=sequence_length),
    layers.SpatialDropout1D(0.2),  # Dropout layer after Embedding
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),  # LSTM layer
    layers.GlobalMaxPooling1D(),  # Global Max Pooling
    layers.Dense(64, activation='relu'),  # Dense layer with ReLU activation
    layers.Dropout(0.5),  # Dropout layer for regularization
    layers.Dense(1, activation='sigmoid')  # Output layer with Sigmoid activation
])

model.summary()

# Compile the Model
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7)]
)

# Train the Model
epochs = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Evaluate the Model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Plot Training and Validation Metrics
history_dict = history.history
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# Export the Model for Inference
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=['accuracy']
)

# Evaluate the Exported Model with Raw Test Data
loss, accuracy = export_model.evaluate(raw_test_ds)
print("Export Model Accuracy: ", accuracy)

# Test the Exported Model with New Examples
examples = tf.constant([
    "very bad",
    "i love it",
    "waste of time",
    "awful movie",
    "amazing movie",
    "I wish I never gave it a try",
    "bad",
    "good",
    "not my cup of tea",
    "could have been better",
    "I want my money back",
    "i enjoyed it",
    "great time",
    "I find it boring",
    "the settings were amazing, the story as well",
    "the actress was not that good"
])

predictions = export_model.predict(examples)
print(predictions)

# Determine Sentiment Labels Based on Predictions
threshold = 0.7
labels = ['positive' if pred > threshold else 'negative' for pred in predictions]

# Print the Results
for example, label in zip(examples.numpy(), labels):
    print(f"Text: {example.decode('utf-8')}, Sentiment: {label}")

# Save the Model
export_model.save('sentiment_model')

# Zip and Download the Model
from google.colab import files
shutil.make_archive('sentiment_model', 'zip', 'sentiment_model')
files.download('sentiment_model.zip')
