# data.py

# Import the required libraries and frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import re
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Load the data sets of scientific questions and phenomena
# The data sets are in JSON format, with the following fields:
# question: the scientific question to be answered or explained
# phenomena: the phenomena of interest to be predicted or explained
# hypotheses: a list of possible hypotheses that can answer or explain the question or the phenomena
# Each hypothesis has the following fields:
# text: the natural language expression of the hypothesis
# logic: the symbolic logic expression of the hypothesis
# math: the mathematical formula expression of the hypothesis
# assumptions: a list of assumptions or constraints of the hypothesis
# confidence: a numerical score indicating the confidence or plausibility of the hypothesis
# For example, a sample data point is:
# {
#     "question": "Why do apples fall from trees?",
#     "phenomena": "The motion of apples falling from trees",
#     "hypotheses": [
#         {
#             "text": "Apples fall from trees because of gravity",
#             "logic": "Gravity(Apple, Earth) -> Fall(Apple)",
#             "math": "F = G * (m1 * m2) / r^2",
#             "assumptions": ["The apple and the earth are spherical masses", "The apple is not attached to the tree by any force"],
#             "confidence": 0.9
#         },
#         {
#             "text": "Apples fall from trees because of wind",
#             "logic": "Wind(Apple) -> Fall(Apple)",
#             "math": "F = 1/2 * rho * v^2 * A * Cd",
#             "assumptions": ["The apple is exposed to the wind", "The wind is strong enough to overcome the apple's inertia"],
#             "confidence": 0.3
#         },
#         {
#             "text": "Apples fall from trees because of entropy",
#             "logic": "Entropy(System) -> Fall(Apple)",
#             "math": "S = k * log(W)",
#             "assumptions": ["The system consists of the apple, the tree, and the earth", "The system is isolated from external influences"],
#             "confidence": 0.1
#         }
#     ]
# }

# Define the paths of the data sets
train_path = "train.json"
test_path = "test.json"

# Define the columns of the data sets
columns = ["question", "phenomena", "hypotheses"]

# Define the subcolumns of the hypotheses field
subcolumns = ["text", "logic", "math", "assumptions", "confidence"]

# Define a function to load and parse the data sets
def load_data(path, columns, subcolumns):
    # Read the JSON file
    with open(path, "r") as f:
        data = json.load(f)
    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data, columns=columns)
    # Expand the hypotheses field into separate columns
    for i, subcolumn in enumerate(subcolumns):
        df[subcolumn] = df["hypotheses"].apply(lambda x: x[i])
    # Drop the hypotheses field
    df = df.drop("hypotheses", axis=1)
    # Return the dataframe
    return df

# Load and parse the train and test data sets
train_df = load_data(train_path, columns, subcolumns)
test_df = load_data(test_path, columns, subcolumns)

# Explore the data sets
# Print the shape and the summary of the data sets
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Train data summary:")
print(train_df.describe())
print("Test data summary:")
print(test_df.describe())

# Print the first five rows of the data sets
print("Train data head:")
print(train_df.head())
print("Test data head:")
print(test_df.head())

# Plot the distribution of the confidence scores of the hypotheses
plt.figure(figsize=(10, 6))
sns.histplot(data=train_df, x="confidence", bins=10, kde=True, label="Train")
sns.histplot(data=test_df, x="confidence", bins=10, kde=True, label="Test")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Distribution of Confidence Scores")
plt.legend()
plt.show()

# Plot the correlation matrix of the numerical features
plt.figure(figsize=(10, 10))
sns.heatmap(data=train_df.corr(), annot=True, square=True, cmap="Blues")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Perform data analysis and visualization
# Define a function to calculate the length of a text
def text_length(text):
    # Return the number of words in the text
    return len(text.split())

# Define a function to calculate the complexity of a logic expression
def logic_complexity(logic):
    # Return the number of symbols in the logic expression
    return len(re.findall("[A-Z()&|~<>=]", logic))

# Define a function to calculate the complexity of a math expression
def math_complexity(math):
    # Return the number of symbols in the math expression
    return len(re.findall("[a-zA-Z0-9()*/+-^=]", math))

# Define a function to calculate the number of assumptions
def assumption_number(assumptions):
    # Return the length of the assumption list
    return len(assumptions)

# Add the new features to the data sets
train_df["question_length"] = train_df["question"].apply(text_length)
train_df["phenomena_length"] = train_df["phenomena"].apply(text_length)
train_df["text_length"] = train_df["text"].apply(text_length)
train_df["logic_complexity"] = train_df["logic"].apply(logic_complexity)
train_df["math_complexity"] = train_df["math"].apply(math_complexity)
train_df["assumption_number"] = train_df["assumptions"].apply(assumption_number)

test_df["question_length"] = test_df["question"].apply(text_length)
test_df["phenomena_length"] = test_df["phenomena"].apply(text_length)
test_df["text_length"] = test_df["text"].apply(text_length)
test_df["logic_complexity"] = test_df["logic"].apply(logic_complexity)
test_df["math_complexity"] = test_df["math"].apply(math_complexity)
test_df["assumption_number"] = test_df["assumptions"].apply(assumption_number)

# Plot the distribution of the new features
plt.figure(figsize=(15, 15))
for i, feature in enumerate(["question_length", "phenomena_length", "text_length", "logic_complexity", "math_complexity", "assumption_number"]):
    plt.subplot(3, 2, i+1)
    sns.histplot(data=train_df, x=feature, bins=10, kde=True, label="Train")
    sns.histplot(data=test_df, x=feature, bins=10, kde=True, label="Test")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title("Distribution of " + feature)
    plt.legend()
plt.tight_layout()
plt.show()

# Plot the correlation matrix of the new features
plt.figure(figsize=(10, 10))
sns.heatmap(data=train_df.corr(), annot=True, square=True, cmap="Blues")
plt.title("Correlation Matrix of New Features")
plt.show()

# Perform data cleaning and preprocessing
# Define a function to remove punctuation and lowercase a text
def clean_text(text):
    # Remove punctuation and lowercase the text
    text = re.sub("[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    # Return the cleaned text
    return text

# Define a function to tokenize and lemmatize a text
def lemmatize_text(text):
    # Initialize the spacy model
    nlp = spacy.load("en_core_web_sm")
    # Tokenize the text
    tokens = nlp(text)
    # Lemmatize the tokens
    lemmas = [token.lemma_ for token in tokens]
    # Join the lemmas
    lemmatized_text = " ".join(lemmas)
    # Return the lemmatized text
    return lemmatized_text

# Apply the clean_text and lemmatize_text functions to the text fields
train_df["question"] = train_df["question"].apply(clean_text).apply(lemmatize_text)
train_df["phenomena"] = train_df["phenomena"].apply(clean_text).apply(lemmatize_text)
train_df["text"] = train_df["text"].apply(clean_text).apply(lemmatize_text)

test_df["question"] = test_df["question"].apply(clean_text).apply(lemmatize_text)
test_df["phenomena"] = test_df["phenomena"].apply(clean_text).apply(lemmatize_text)
test_df["text"] = test_df["text"].apply(clean_text).apply(lemmatize_text)

# Define a function to simplify the logic expressions
def simplify_logic(logic):
    # Replace the symbols with their equivalent operators
    logic = logic.replace("&", "and")
    logic = logic.replace("|", "or")
    logic = logic.replace("~", "not")
    logic = logic.replace("->", "implies")
    logic = logic.replace("<->", "iff")
    # Return the simplified logic
    return logic

# Define a function to simplify the math expressions
def simplify_math(math):
    # Replace the symbols with their equivalent operators
    math = math.replace("*", "times")
    math = math.replace("/", "divided by")
    math = math.replace("+", "plus")
    math = math.replace("-", "minus")
    math = math.replace("^", "to the power of")
    math = math.replace("=", "equals")
    # Return the simplified math
    return math

# Apply the simplify_logic and simplify_math functions to the logic and math fields
train_df["logic"] = train_df["logic"].apply(simplify_logic)
train_df["math"] = train_df["math"].apply(simplify_math)

test_df["logic"] = test_df["logic"].apply(simplify_logic)
test_df["math"] = test_df["math"].apply(simplify_math)

# Define a function to flatten the assumption lists
def flatten_assumptions(assumptions):
    # Join the assumptions with a comma
    flattened_assumptions = ", ".join(assumptions)
    # Return the flattened assumptions
    return flattened_assumptions

# Apply the flatten_assumptions function to the assumptions field
train_df["assumptions"] = train_df["assumptions"].apply(flatten_assumptions)
test_df["assumptions"] = test_df["assumptions"].apply(flatten_assumptions)

# Perform data transformation and encoding
# Define the input and output features
input_features = ["question", "phenomena", "text", "logic", "math", "assumptions", "question_length", "phenomena_length", "text_length", "logic_complexity", "math_complexity", "assumption_number"]
output_feature = "confidence"

# Split the data sets into input and output
X_train = train_df[input_features]
y_train = train_df[output_feature]
X_test = test_df[input_features]
y_test = test_df[output_feature]

# Vectorize the text fields using TF-IDF
tfidf = TfidfVectorizer()
X_train_text = tfidf.fit_transform(X_train["question"] + " " + X_train["phenomena"] + " " + X_train["text"] + " " + X_train["logic"] + " " + X_train["math"] + " " + X_train["assumptions"])
X_test_text = tfidf.transform(X_test["question"] + " " + X_test["phenomena"] + " " + X_test["text"] + " " + X_test["logic"] + " " + X_test["math"] + " " + X_test["assumptions"])

# Scale the numerical fields using standard scaler
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[["question_length", "phenomena_length", "text_length", "logic_complexity", "math_complexity", "assumption_number"]])
X_test_num = scaler.transform(X_test[["question_length", "phenomena_length", "text_length", "logic_complexity", "math_complexity", "assumption_number"]])

# Concatenate the text and numerical features
X_train = np.concatenate((X_train_text.toarray(), X_train_num), axis=1)
X_test = np.concatenate((X_test_text.toarray(), X_test_num), axis=1)

# Save the data sets as numpy arrays
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
