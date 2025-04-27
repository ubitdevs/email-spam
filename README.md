# 📄 Email Spam Classifier — Documentation

# 1. Project Overview
This project develops an Email Spam Classifier that identifies whether a given email message is spam or not. The model uses natural language processing (NLP) techniques for preprocessing, various vectorization strategies for feature extraction, and machine learning for classification.

# 2. Libraries and Tools Used

pandas — data loading and manipulation

numpy — numerical computations

nltk — natural language processing (tokenization, stopwords, lemmatization)

scikit-learn — model building, vectorization, evaluation

imbalanced-learn (imblearn) — handling class imbalance via RandomOverSampler

gensim — Word2Vec embeddings

matplotlib & seaborn — data visualization

joblib — model saving

# 3. Data Preparation
 
Dataset: spam.csv

Columns selected:

message: the text content of the email/SMS.

label: the target class ("ham" for legitimate and "spam" for spam).

Renamed columns for clarity.

# 4. Text Preprocessing

Tokenization: Split the text into words.

Stopwords Removal: Removed common English stopwords (like "the", "and").

Lemmatization: Reduced words to their base form (e.g., "running" → "run").

Regular Expressions (Regex): Used for cleaning unwanted characters from text.

# 5. Feature Extraction
Three different techniques were explored:

Count Vectorization: Converts text to a bag-of-words representation.

TF-IDF Vectorization: Calculates importance of words across the dataset.

Word2Vec Embedding: Captures semantic meaning by representing words as dense vectors.

Normalization was applied using MinMaxScaler to ensure features were scaled properly.

# 6. Handling Imbalanced Data
The dataset had more "ham" messages than "spam."

RandomOverSampler was used to balance the dataset by oversampling the minority class.

# 7. Model Building

Logistic Regression was selected as the classification model.

A pipeline was created combining:

Vectorization

Scaling (if necessary)

Logistic Regression

Train-test split: Data was split into training and testing sets.

# 8. Model Evaluation

Confusion Matrix and Classification Report were generated.

Metrics calculated:

Precision

Recall

F1-score

Accuracy

Visualization:

A heatmap was created for the confusion matrix using seaborn.

# 9. Model Saving

The trained model pipeline was saved as a .pkl file using joblib for future use without retraining.

# 10. Summary

This project successfully built a functional, efficient, and interpretable spam classifier with good performance metrics.
The use of different feature extraction methods and data balancing techniques significantly improved model accuracy.

