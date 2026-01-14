"""
=================================================================
SPAM MESSAGE/EMAIL DETECTION USING MACHINE LEARNING
A Beginner-Friendly Project
=================================================================
"""

# =================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =================================================================
import pandas as pd
import numpy as np
import re
import string

# For text preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# For evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score

# For stopwords removal
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)


# =================================================================
# STEP 2: LOAD AND EXPLORE THE DATASET
# =================================================================
def load_dataset():
    """
    Load the spam dataset.
    
    For this example, we'll create a sample dataset.
    In real projects, you can use datasets like:
    - SMS Spam Collection Dataset (from UCI ML Repository)
    - Kaggle spam email datasets
    """
    
    # Sample dataset (In real project, use pd.read_csv('spam.csv'))
    data = {
        'message': [
            'Congratulations! You have won a $1000 gift card. Click here to claim now!',
            'Hey, are we still meeting for lunch tomorrow?',
            'URGENT: Your account will be suspended. Verify your identity immediately!',
            'Can you send me the project report by evening?',
            'Get rich quick! Earn $5000 per week working from home!',
            'Happy birthday! Hope you have a wonderful day!',
            'Free entry to win iPhone 14! Text WIN to 12345',
            'Meeting rescheduled to 3 PM. Please confirm.',
            'You are a winner! Claim your prize money now!!!',
            'Let me know if you need any help with the assignment',
            'CLICK HERE for FREE credit card approval in 5 minutes',
            'Thanks for your help yesterday, really appreciate it',
            'Limited time offer! Buy 1 get 1 free on all products',
            'Could you please review the document I sent?',
            'Call now to claim your lottery winnings of $50000',
            'See you at the conference next week',
            'Your loan has been approved! No credit check required!',
            'What time does the movie start tonight?',
            'Act now! Exclusive discount ending soon! Visit our website',
            'Reminder: Team meeting at 10 AM tomorrow'
        ],
        'label': [
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham'
        ]
    }
    
    df = pd.DataFrame(data)
    
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Total messages: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print("\nSample messages:")
    print(df.head())
    print("="*60)
    
    return df


# =================================================================
# STEP 3: TEXT PREPROCESSING
# =================================================================
def preprocess_text(text):
    """
    Clean and preprocess the text data.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation and numbers
    3. Remove stopwords
    4. Tokenize (split into words)
    
    Args:
        text (str): Input message
        
    Returns:
        str: Cleaned text
    """
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Step 3: Tokenization (split into words)
    words = text.split()
    
    # Step 4: Remove stopwords (common words like 'the', 'is', 'are')
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text


def preprocess_dataset(df):
    """
    Apply preprocessing to entire dataset.
    
    Args:
        df: DataFrame with 'message' column
        
    Returns:
        df: DataFrame with cleaned messages
    """
    print("\n" + "="*60)
    print("PREPROCESSING TEXT DATA")
    print("="*60)
    
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    
    print("Example of preprocessing:")
    print(f"\nOriginal: {df['message'].iloc[0]}")
    print(f"Cleaned:  {df['cleaned_message'].iloc[0]}")
    print("="*60)
    
    return df


# =================================================================
# STEP 4: CONVERT TEXT TO NUMERICAL FEATURES (TF-IDF)
# =================================================================
def vectorize_text(X_train, X_test):
    """
    Convert text into numerical features using TF-IDF.
    
    TF-IDF (Term Frequency - Inverse Document Frequency):
    - Measures importance of words in documents
    - Common words get lower scores
    - Rare, distinctive words get higher scores
    
    Args:
        X_train: Training messages
        X_test: Testing messages
        
    Returns:
        X_train_vec, X_test_vec, vectorizer
    """
    print("\n" + "="*60)
    print("CONVERTING TEXT TO NUMERICAL FEATURES (TF-IDF)")
    print("="*60)
    
    # Create TF-IDF vectorizer
    # max_features: Use only top 500 most frequent words
    vectorizer = TfidfVectorizer(max_features=500)
    
    # Fit on training data and transform both train and test
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Training data shape: {X_train_vec.shape}")
    print(f"Testing data shape: {X_test_vec.shape}")
    print(f"Number of features (unique words): {len(vectorizer.get_feature_names_out())}")
    print("="*60)
    
    return X_train_vec, X_test_vec, vectorizer


# =================================================================
# STEP 5: TRAIN THE NAIVE BAYES MODEL
# =================================================================
def train_model(X_train, y_train):
    """
    Train Naive Bayes classifier.
    
    Multinomial Naive Bayes:
    - Works well for text classification
    - Based on probability theory (Bayes' Theorem)
    - Assumes features (words) are independent
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        model: Trained classifier
    """
    print("\n" + "="*60)
    print("TRAINING NAIVE BAYES MODEL")
    print("="*60)
    
    # Create and train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("Model training completed successfully!")
    print("="*60)
    
    return model


# =================================================================
# STEP 6: EVALUATE THE MODEL
# =================================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using multiple metrics.
    
    Metrics:
    - Accuracy: Overall correctness
    - Precision: Of predicted spam, how many are actually spam
    - Recall: Of actual spam, how many did we catch
    - Confusion Matrix: Detailed breakdown of predictions
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    
    # Confusion Matrix
    print("\n" + "-"*60)
    print("CONFUSION MATRIX")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    print(f"\n                Predicted")
    print(f"              Ham    Spam")
    print(f"Actual Ham    {cm[0][0]}      {cm[0][1]}")
    print(f"       Spam   {cm[1][0]}      {cm[1][1]}")
    
    # Detailed classification report
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*60)
    print(classification_report(y_test, y_pred))
    print("="*60)


# =================================================================
# STEP 7: PREDICT NEW MESSAGES
# =================================================================
def predict_message(message, model, vectorizer):
    """
    Predict whether a new message is spam or ham.
    
    Args:
        message (str): New message to classify
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        str: Prediction ('spam' or 'ham')
    """
    # Preprocess the message
    cleaned_message = preprocess_text(message)
    
    # Convert to numerical features
    message_vec = vectorizer.transform([cleaned_message])
    
    # Make prediction
    prediction = model.predict(message_vec)[0]
    
    # Get probability scores
    proba = model.predict_proba(message_vec)[0]
    
    return prediction, proba


# =================================================================
# MAIN EXECUTION
# =================================================================
def main():
    """
    Main function to execute the complete pipeline.
    """
    print("\n")
    print("*"*60)
    print("  SPAM EMAIL/MESSAGE DETECTION USING MACHINE LEARNING")
    print("*"*60)
    
    # Step 1: Load dataset
    df = load_dataset()
    
    # Step 2: Preprocess text
    df = preprocess_dataset(df)
    
    # Step 3: Split data into train and test sets (80-20 split)
    print("\n" + "="*60)
    print("SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("="*60)
    X = df['cleaned_message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print("="*60)
    
    # Step 4: Vectorize text (TF-IDF)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    # Step 5: Train model
    model = train_model(X_train_vec, y_train)
    
    # Step 6: Evaluate model
    evaluate_model(model, X_test_vec, y_test)
    
    # Step 7: Test with new messages
    print("\n" + "="*60)
    print("TESTING WITH NEW MESSAGES")
    print("="*60)
    
    test_messages = [
        "Congratulations! You've won a free iPhone. Click now!",
        "Can we discuss the project tomorrow at 2 PM?",
        "URGENT: Your account needs verification immediately!"
    ]
    
    for msg in test_messages:
        prediction, proba = predict_message(msg, model, vectorizer)
        print(f"\nMessage: {msg}")
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: Ham={proba[0]:.2%}, Spam={proba[1]:.2%}")
        print("-"*60)
    
    print("\n" + "*"*60)
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("*"*60 + "\n")
    
    return model, vectorizer


# =================================================================
# RUN THE PROJECT
# =================================================================
if __name__ == "__main__":
    model, vectorizer = main()
    
    # Interactive prediction function
    print("\n" + "="*60)
    print("You can now test your own messages!")
    print("="*60)
    print("\nExample usage:")
    print(">>> message = 'Win lottery money now!!!'")
    print(">>> prediction, proba = predict_message(message, model, vectorizer)")
    print(">>> print(f'Prediction: {prediction}')")