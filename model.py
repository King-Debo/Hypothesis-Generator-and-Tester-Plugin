# model.py

# Import the required libraries and frameworks
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu

# Load the data sets from numpy arrays
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Define the input and output dimensions
input_dim = X_train.shape[1]
output_dim = 1

# Define the model architecture
# The model consists of three parts: encoder, decoder, and regressor
# The encoder encodes the input features into a latent vector
# The decoder decodes the latent vector into a natural language hypothesis
# The regressor predicts the confidence score of the hypothesis

# Define the encoder
encoder_input = layers.Input(shape=(input_dim,))
encoder_output = layers.Dense(256, activation="relu")(encoder_input)
encoder_output = layers.Dropout(0.2)(encoder_output)
encoder_output = layers.Dense(128, activation="relu")(encoder_output)
encoder_output = layers.Dropout(0.2)(encoder_output)
encoder_output = layers.Dense(64, activation="relu")(encoder_output)
encoder_output = layers.Dropout(0.2)(encoder_output)
encoder = models.Model(encoder_input, encoder_output, name="encoder")

# Define the decoder
decoder_input = layers.Input(shape=(64,))
decoder_output = layers.RepeatVector(20)(decoder_input)
decoder_output = layers.LSTM(64, return_sequences=True)(decoder_output)
decoder_output = layers.TimeDistributed(layers.Dense(10000, activation="softmax"))(decoder_output)
decoder = models.Model(decoder_input, decoder_output, name="decoder")

# Define the regressor
regressor_input = layers.Input(shape=(64,))
regressor_output = layers.Dense(32, activation="relu")(regressor_input)
regressor_output = layers.Dropout(0.2)(regressor_output)
regressor_output = layers.Dense(16, activation="relu")(regressor_output)
regressor_output = layers.Dropout(0.2)(regressor_output)
regressor_output = layers.Dense(output_dim, activation="sigmoid")(regressor_output)
regressor = models.Model(regressor_input, regressor_output, name="regressor")

# Define the model
model_input = layers.Input(shape=(input_dim,))
model_output_1 = encoder(model_input)
model_output_2 = decoder(model_output_1)
model_output_3 = regressor(model_output_1)
model = models.Model(model_input, [model_output_2, model_output_3], name="model")

# Define the model parameters and hyperparameters
optimizer = optimizers.Adam(learning_rate=0.001)
loss_1 = losses.SparseCategoricalCrossentropy()
loss_2 = losses.MeanSquaredError()
metric_1 = metrics.SparseCategoricalAccuracy()
metric_2 = metrics.RootMeanSquaredError()
epochs = 100
batch_size = 32
patience = 10
factor = 0.1
min_lr = 0.00001

# Compile the model
model.compile(optimizer=optimizer, loss=[loss_1, loss_2], metrics=[metric_1, metric_2])

# Define the model callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=patience//2, min_lr=min_lr)

# Train the model
model.fit(X_train, [y_train, y_train], epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Test the model
model.evaluate(X_test, [y_test, y_test], batch_size=batch_size)

# Define a function to generate a hypothesis from a latent vector
def generate_hypothesis(latent_vector):
    # Decode the latent vector into a sequence of word indices
    word_indices = decoder.predict(latent_vector.reshape(1, -1))[0]
    # Convert the word indices into words using the inverse vocabulary
    inverse_vocab = {v: k for k, v in tfidf.vocabulary_.items()}
    words = [inverse_vocab.get(i, "<UNK>") for i in word_indices]
    # Join the words into a hypothesis
    hypothesis = " ".join(words)
    # Return the hypothesis
    return hypothesis

# Define a function to calculate the BLEU score of a hypothesis
def bleu_score(hypothesis, reference):
    # Tokenize the hypothesis and the reference
    hypothesis_tokens = hypothesis.split()
    reference_tokens = reference.split()
    # Calculate the BLEU score
    score = sentence_bleu([reference_tokens], hypothesis_tokens)
    # Return the score
    return score
