# deploy.py

# Import the required libraries and frameworks
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu
import streamlit as st
import requests

# Load the data sets from numpy arrays
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load the best model from the tune.py file
best_model = keras.models.load_model("model.h5")

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

# Deploy and maintain the hypothesis generator and tester plugin
# Use the streamlit framework to create a web interface for the plugin
st.title("Hypothesis Generator and Tester Plugin")
st.write("This plugin can generate and test hypotheses for scientific questions and phenomena. It can also provide confidence scores for the hypotheses.")

# Create a sidebar for the user inputs
st.sidebar.header("User Inputs")
# Create a text input for the scientific question
question = st.sidebar.text_input("Enter a scientific question:")
# Create a text input for the phenomena of interest
phenomena = st.sidebar.text_input("Enter the phenomena of interest:")
# Create a slider for the number of hypotheses to generate
num_hypotheses = st.sidebar.slider("Select the number of hypotheses to generate:", 1, 5, 1)

# Create a button for the user to submit the inputs
submit = st.sidebar.button("Submit")

# If the user submits the inputs, process them and display the outputs
if submit:
    # Validate the inputs
    if question == "" or phenomena == "":
        # Display an error message
        st.error("Please enter a valid question and phenomena.")
    else:
        # Display a success message
        st.success("Your inputs have been submitted.")
        # Preprocess the inputs
        question = clean_text(question)
        question = lemmatize_text(question)
        phenomena = clean_text(phenomena)
        phenomena = lemmatize_text(phenomena)
        # Vectorize the inputs
        input_text = tfidf.transform([question + " " + phenomena])
        # Predict the hypotheses and the confidence scores
        hypotheses, confidences = best_model.predict(input_text)
        # Convert the hypotheses from word indices to words
        hypotheses = [generate_hypothesis(hypothesis) for hypothesis in hypotheses]
        # Display the hypotheses and the confidence scores
        st.header("Hypotheses and Confidence Scores")
        for i in range(num_hypotheses):
            st.write(f"Hypothesis {i+1}: {hypotheses[i]}")
            st.write(f"Confidence Score: {confidences[i]}")
        # Create a selectbox for the user to choose a hypothesis to test
        st.header("Hypothesis Testing")
        hypothesis_to_test = st.selectbox("Select a hypothesis to test:", hypotheses)
        # Create a text input for the user to enter the test data
        test_data = st.text_input("Enter the test data:")
        # Create a button for the user to submit the test data
        test = st.button("Test")
        # If the user submits the test data, process it and display the output
        if test:
            # Validate the test data
            if test_data == "":
                # Display an error message
                st.error("Please enter a valid test data.")
            else:
                # Display a success message
                st.success("Your test data has been submitted.")
                # Preprocess the test data
                test_data = clean_text(test_data)
                test_data = lemmatize_text(test_data)
                # Vectorize the test data
                test_text = tfidf.transform([test_data])
                # Predict the confidence score for the hypothesis given the test data
                confidence = best_model.predict(test_text)[1][0]
                # Display the confidence score
                st.write(f"Confidence Score: {confidence}")
                # Calculate the BLEU score for the hypothesis given the test data
                bleu = bleu_score(hypothesis_to_test, test_data)
                # Display the BLEU score
                st.write(f"BLEU Score: {bleu}")

# Create a feedback system for the user to rate the plugin
st.header("Feedback")
st.write("Please rate the plugin based on your experience.")
# Create a slider for the user to rate the plugin from 1 to 5 stars
rating = st.slider("Rating:", 1, 5, 3)
# Create a text input for the user to enter the feedback comments
comments = st.text_input("Comments:")
# Create a button for the user to submit the feedback
feedback = st.button("Submit Feedback")
# If the user submits the feedback, process it and display the output
if feedback:
    # Validate the feedback
    if rating == "" or comments == "":
        # Display an error message
        st.error("Please enter a valid rating and comments.")
    else:
        # Display a success message
        st.success("Your feedback has been submitted.")
        # Save the feedback to a file
        with open("feedback.txt", "a") as f:
            f.write(f"Rating: {rating}\n")
            f.write(f"Comments: {comments}\n")
            f.write("\n")

# Update and improve the plugin
# Define a function to retrain the plugin with the new feedback data
def retrain_plugin(feedback_data):
    # Load the feedback data from the file
    feedback_df = pd.read_csv(feedback_data, sep="\t")
    # Preprocess the feedback data
    feedback_df["question"] = feedback_df["question"].apply(clean_text).apply(lemmatize_text)
    feedback_df["phenomena"] = feedback_df["phenomena"].apply(clean_text).apply(lemmatize_text)
    feedback_df["hypothesis"] = feedback_df["hypothesis"].apply(clean_text).apply(lemmatize_text)
    # Vectorize the feedback data
    feedback_text = tfidf.transform(feedback_df["question"] + " " + feedback_df["phenomena"] + " " + feedback_df["hypothesis"])
    feedback_confidence = feedback_df["rating"] / 5
    # Concatenate the feedback data with the original data
    X_train_new = np.concatenate((X_train, feedback_text.toarray()), axis=0)
    y_train_new = np.concatenate((y_train, feedback_confidence), axis=0)
    # Retrain the best model with the new data
    best_model.fit(X_train_new, [y_train_new, y_train_new], epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])
    # Retest the best model with the test data
    best_model.evaluate(X_test, [y_test, y_test], batch_size=batch_size)
    # Return the retrained model
    return best_model

# Define a function to retest the plugin with the new test data
def retest_plugin(test_data):
    # Load the test data from the file
    test_df = pd.read_csv(test_data, sep="\t")
    # Preprocess the test data
    test_df["question"] = test_df["question"].apply(clean_text).apply(lemmatize_text)
    test_df["phenomena"] = test_df["phenomena"].apply(clean_text).apply(lemmatize_text)
    test_df["hypothesis"] = test_df["hypothesis"].apply(clean_text).apply(lemmatize_text)
    # Vectorize the test data
    test_text = tfidf.transform(test_df["question"] + " " + test_df["phenomena"] + " " + test_df["hypothesis"])
    test_confidence = test_df["rating"] / 5
    # Retest the best model with the new test data
    best_model.evaluate(test_text, [test_confidence, test_confidence], batch_size=batch_size)
