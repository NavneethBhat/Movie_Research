import pandas as pd
import joblib # For loading models
from sklearn.feature_extraction.text import TfidfVectorizer # Even though loaded, needed for type hint/context
from sklearn.model_selection import train_test_split # To get X_test, y_test from the original df
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report # New imports for detailed evaluation
import matplotlib.pyplot as plt # For plotting confusion matrix

# --- Configuration for file paths ---
file_path_input_df = 'synthetic_movie_reviews_text_processed.csv'
model_input_path = 'logistic_regression_sentiment_model.joblib'
vectorizer_input_path = 'tfidf_vectorizer.joblib'

try:
    print("--- Starting Phase 5: Model Evaluation and Optimization ---")

    # --- 1. Load the Saved Model and TF-IDF Vectorizer ---
    print("\n--- Loading Saved Model and Vectorizer ---")
    loaded_model = joblib.load(model_input_path)
    loaded_vectorizer = joblib.load(vectorizer_input_path)
    print(f"Model loaded from: {model_input_path}")
    print(f"Vectorizer loaded from: {vectorizer_input_path}")

    # --- 2. Load the text processed DataFrame again to get original X and y for splitting ---
    # This ensures we get the same test set as during training.
    df = pd.read_csv(file_path_input_df)
    print(f"\nDataFrame loaded from {file_path_input_df} for evaluation.")

    # Re-vectorize the 'processed_review' column using the loaded vectorizer
    # It's crucial to use the *loaded* vectorizer to transform new/existing data
    # into the exact feature space the model was trained on.
    X_all_features = loaded_vectorizer.transform(df['processed_review'])
    y_all_sentiment = df['Sentiment']

    # Re-split the data to get the exact X_test and y_test used during training
    # Ensure random_state and test_size match Phase 4 exactly.
    _, X_test, _, y_test = train_test_split(X_all_features, y_all_sentiment,
                                            test_size=0.2, random_state=42, stratify=y_all_sentiment)
    print(f"Re-created X_test shape: {X_test.shape}")
    print(f"Re-created y_test shape: {y_test.shape}")


    # --- 3. Make Predictions on the Test Set using the Loaded Model ---
    print("\n--- Making Predictions with Loaded Model ---")
    y_pred_loaded = loaded_model.predict(X_test)
    print("Predictions made successfully.")

    # --- 4. In-depth Evaluation: Confusion Matrix and Classification Report ---
    print("\n--- Detailed Model Evaluation ---")

    # Classification Report (re-display for full context)
    print("\nClassification Report (from Loaded Model):")
    print(classification_report(y_test, y_pred_loaded, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_loaded, labels=loaded_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=loaded_model.classes_)

    print("\nDisplaying Confusion Matrix:")
    plt.figure(figsize=(10, 8)) # Adjust figure size as needed
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca()) # Pass ax=plt.gca() to plot on current figure
    plt.title('Confusion Matrix for Logistic Regression Sentiment Model')
    plt.show()

    print("\n--- Phase 5: Model Evaluation Step 1 Completed ---")

except FileNotFoundError as e:
    print(f"Error: A required file was not found. Please ensure all input files ({file_path_input_df}, {model_input_path}, {vectorizer_input_path}) exist in your environment. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred during Phase 5 evaluation: {e}")
