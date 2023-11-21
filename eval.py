# eval.py

# Import the required libraries and frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, r2_score
from scipy.stats import chi2_contingency

# Load the data sets from numpy arrays
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load the model from the model.py file
from model import model, generate_hypothesis, bleu_score

# Predict the hypotheses and the confidence scores for the test data
y_pred_text, y_pred_conf = model.predict(X_test)

# Convert the predicted hypotheses from word indices to words
inverse_vocab = {v: k for k, v in tfidf.vocabulary_.items()}
y_pred_text = [" ".join([inverse_vocab.get(i, "<UNK>") for i in y_pred_text[j]]) for j in range(len(y_pred_text))]

# Convert the test data from numpy arrays to pandas dataframes
test_df = pd.DataFrame(X_test, columns=input_features)
test_df["confidence"] = y_test
test_df["question"] = test_df["question"].apply(clean_text).apply(lemmatize_text)
test_df["phenomena"] = test_df["phenomena"].apply(clean_text).apply(lemmatize_text)
test_df["text"] = test_df["text"].apply(clean_text).apply(lemmatize_text)
test_df["logic"] = test_df["logic"].apply(simplify_logic)
test_df["math"] = test_df["math"].apply(simplify_math)
test_df["assumptions"] = test_df["assumptions"].apply(flatten_assumptions)

# Add the predicted hypotheses and the confidence scores to the test dataframe
test_df["pred_text"] = y_pred_text
test_df["pred_conf"] = y_pred_conf

# Evaluate and compare the results and the performance of the models and algorithms
# Define the evaluation criteria and the comparison metrics
criteria = ["accuracy", "simplicity", "consistency", "falsifiability", "generality"]
metrics = ["precision", "recall", "f1_score", "mse", "r2_score", "likelihood_ratio", "bayesian_factor", "information_criterion"]

# Define a function to calculate the accuracy of a hypothesis
def accuracy(hypothesis, reference):
    # Return 1 if the hypothesis is the same as the reference, 0 otherwise
    return int(hypothesis == reference)

# Define a function to calculate the simplicity of a hypothesis
def simplicity(hypothesis):
    # Return the inverse of the length of the hypothesis
    return 1 / len(hypothesis.split())

# Define a function to calculate the consistency of a hypothesis
def consistency(hypothesis, question, phenomena):
    # Return 1 if the hypothesis is consistent with the question and the phenomena, 0 otherwise
    # For simplicity, we assume that the hypothesis is consistent if it contains the same keywords as the question and the phenomena
    question_keywords = set(question.split())
    phenomena_keywords = set(phenomena.split())
    hypothesis_keywords = set(hypothesis.split())
    return int(question_keywords.issubset(hypothesis_keywords) and phenomena_keywords.issubset(hypothesis_keywords))

# Define a function to calculate the falsifiability of a hypothesis
def falsifiability(hypothesis):
    # Return 1 if the hypothesis is falsifiable, 0 otherwise
    # For simplicity, we assume that the hypothesis is falsifiable if it contains a conditional or a causal statement
    return int("if" in hypothesis or "cause" in hypothesis or "effect" in hypothesis)

# Define a function to calculate the generality of a hypothesis
def generality(hypothesis):
    # Return 1 if the hypothesis is general, 0 otherwise
    # For simplicity, we assume that the hypothesis is general if it contains a universal quantifier
    return int("all" in hypothesis or "every" in hypothesis or "any" in hypothesis)

# Define a function to calculate the likelihood ratio of a hypothesis
def likelihood_ratio(hypothesis, data):
    # Return the ratio of the probability of the data given the hypothesis to the probability of the data given the null hypothesis
    # For simplicity, we assume that the probability of the data given the hypothesis is the confidence score of the hypothesis
    # and the probability of the data given the null hypothesis is 0.5
    return hypothesis["confidence"] / 0.5

# Define a function to calculate the Bayesian factor of a hypothesis
def bayesian_factor(hypothesis, data):
    # Return the ratio of the posterior probability of the hypothesis given the data to the prior probability of the hypothesis
    # For simplicity, we assume that the posterior probability of the hypothesis given the data is the confidence score of the hypothesis
    # and the prior probability of the hypothesis is 0.5
    return hypothesis["confidence"] / 0.5

# Define a function to calculate the information criterion of a hypothesis
def information_criterion(hypothesis, data):
    # Return the negative log-likelihood of the data given the hypothesis plus a penalty term for the complexity of the hypothesis
    # For simplicity, we assume that the negative log-likelihood of the data given the hypothesis is the inverse of the confidence score of the hypothesis
    # and the penalty term for the complexity of the hypothesis is the length of the hypothesis
    return -np.log(hypothesis["confidence"]) + len(hypothesis["text"].split())

# Calculate the evaluation criteria and the comparison metrics for the test data
for criterion in criteria:
    test_df[criterion] = test_df.apply(lambda x: eval(criterion)(x["pred_text"], x["text"], x["question"], x["phenomena"]), axis=1)
for metric in metrics:
    test_df[metric] = test_df.apply(lambda x: eval(metric)(x["pred_conf"], x["confidence"]), axis=1)

# Print the summary of the evaluation criteria and the comparison metrics
print("Evaluation summary:")
print(test_df.describe())

# Visualize and analyze the results and the performance of the models and algorithms
# Plot the distribution of the evaluation criteria and the comparison metrics
plt.figure(figsize=(15, 15))
for i, column in enumerate(criteria + metrics):
    plt.subplot(4, 3, i+1)
    sns.histplot(data=test_df, x=column, bins=10, kde=True)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title("Distribution of " + column)
plt.tight_layout()
plt.show()

# Plot the scatter plot of the predicted confidence scores versus the actual confidence scores
plt.figure(figsize=(10, 10))
sns.scatterplot(data=test_df, x="pred_conf", y="confidence")
plt.xlabel("Predicted Confidence Score")
plt.ylabel("Actual Confidence Score")
plt.title("Predicted Confidence Score vs Actual Confidence Score")
plt.show()

# Plot the heat map of the evaluation criteria and the comparison metrics
plt.figure(figsize=(10, 10))
sns.heatmap(data=test_df.corr(), annot=True, square=True, cmap="Blues")
plt.title("Heat Map of Evaluation Criteria and Comparison Metrics")
plt.show()
