import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib # Required for saving/loading models

# --- Configuration for file paths ---
# Input file from Phase 3 (text processed data)
file_path_input = 'synthetic_movie_reviews_text_processed.csv'
# Output files for saving the trained model and vectorizer
model_output_path = 'logistic_regression_sentiment_model.joblib'
vectorizer_output_path = 'tfidf_vectorizer.joblib'

try:
    print("--- Starting Phase 4: Model Building & Training ---")

    # --- 1. Load the text processed DataFrame (from Phase 3 output) ---
    df = pd.read_csv(file_path_input)
    print(f"\nDataFrame loaded from {file_path_input}")
    print("Sample of 'processed_review' column:")
    print(df['processed_review'].head())

    # --- 2. Feature Extraction: Initialize and Fit TF-IDF Vectorizer ---
    print("\n--- Performing TF-IDF Feature Extraction ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    tfidf_features = tfidf_vectorizer.fit_transform(df['processed_review'])
    print(f"TF-IDF Feature Matrix created with shape: {tfidf_features.shape}")

    # --- 3. Define Features (X) and Target (y) ---
    X = tfidf_features # Our features are the TF-IDF vectors
    y = df['Sentiment'] # Our target variable is the 'Sentiment' column

    # --- 4. Split the data into training and testing sets ---
    print("\n--- Splitting Data into Training and Testing Sets ---")
    # stratify=y ensures that the proportion of sentiment classes is the same in both train and test sets.
    # random_state for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Data split successful:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nSentiment distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nSentiment distribution in testing set:")
    print(y_test.value_counts(normalize=True))


    # --- 5. Model Selection & 6. Model Training (Logistic Regression) ---
    print("\n--- Training Logistic Regression Model ---")
    # max_iter increased for convergence with sparse features common in text data
    # solver='saga' handles multi_class='auto' and L1/L2 regularization well for multiclass problems.
    # The FutureWarning about 'multi_class' is expected but harmless for now; 'multinomial' will be the default.
    log_reg_model = LogisticRegression(max_iter=1000, solver='saga', multi_class='auto', random_state=42)
    log_reg_model.fit(X_train, y_train)
    print("Logistic Regression Model trained successfully.")

    # --- 7. Make Predictions on the Test Set ---
    y_pred = log_reg_model.predict(X_test)
    print("\nPredictions made on the test set.")

    # --- 8. Initial Model Evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n--- Logistic Regression Model Performance on Test Set ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # --- 9. Save the Trained Model and TF-IDF Vectorizer ---
    print("\n--- Saving Model and Vectorizer ---")
    joblib.dump(log_reg_model, model_output_path)
    joblib.dump(tfidf_vectorizer, vectorizer_output_path)
    print(f"Model saved to: {model_output_path}")
    print(f"Vectorizer saved to: {vectorizer_output_path}")
    print("\n--- Phase 4: Model Building & Training Completed ---")

except FileNotFoundError:
    print(f"Error: The input file '{file_path_input}' was not found. Please ensure '{file_path_input}' exists in your VS Code environment.")
except Exception as e:
    print(f"An unexpected error occurred during Phase 4: {e}")
