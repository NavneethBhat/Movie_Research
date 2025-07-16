import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV # New import for GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Configuration for file paths ---
file_path_input = 'synthetic_movie_reviews_text_processed.csv'
# Optional: save best model and vectorizer from tuning
tuned_model_output_path = 'tuned_logistic_regression_sentiment_model.joblib'
tuned_vectorizer_output_path = 'tfidf_vectorizer_for_tuned_model.joblib'


try:
    print("--- Starting Phase 5: Model Evaluation and Optimization (Hyperparameter Tuning) ---")

    # --- 1. Load the text processed DataFrame ---
    df = pd.read_csv(file_path_input)
    print(f"\nDataFrame loaded from {file_path_input}")

    # --- 2. Feature Extraction: Initialize and Fit TF-IDF Vectorizer ---
    # We will use the same vectorizer parameters as before for now,
    # but these could also be tuned in a more complex pipeline.
    print("\n--- Performing TF-IDF Feature Extraction ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    tfidf_features = tfidf_vectorizer.fit_transform(df['processed_review'])
    print(f"TF-IDF Feature Matrix created with shape: {tfidf_features.shape}")

    # --- 3. Define Features (X) and Target (y) ---
    X = tfidf_features
    y = df['Sentiment']

    # --- 4. Split the data into training and testing sets ---
    print("\n--- Splitting Data into Training and Testing Sets ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split successful: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # --- 5. Hyperparameter Tuning using GridSearchCV ---
    print("\n--- Starting Hyperparameter Tuning for Logistic Regression (using GridSearchCV) ---")

    # Define the parameter grid to search
    # We will tune the 'C' parameter (inverse of regularization strength)
    # Smaller C means stronger regularization.
    param_grid = {
        'C': [0.1, 1, 10, 100], # Common range for C
        # 'penalty': ['l1', 'l2'], # Can also tune penalty, but requires different solvers
        # 'solver': ['liblinear', 'saga'] # Compatible solvers for l1/l2
    }

    # Initialize the Logistic Regression model (with a compatible solver for L2 regularization and auto multi_class)
    # Note: 'saga' solver supports both L1 and L2 penalties and is good for multiclass problems.
    # The FutureWarning about 'multi_class' is expected but harmless for now.
    log_reg = LogisticRegression(max_iter=1000, solver='saga', multi_class='auto', random_state=42)

    # Initialize GridSearchCV
    # cv=5 means 5-fold cross-validation
    # scoring='accuracy' is a common metric; can also use 'f1_weighted' for imbalanced classes
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    # n_jobs=-1 uses all available CPU cores for faster computation
    # verbose=2 shows progress during tuning

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # --- 6. Evaluate the Best Model found by GridSearchCV ---
    best_log_reg_model = grid_search.best_estimator_
    print("\n--- Evaluating the Best Tuned Model on the Test Set ---")

    y_pred_tuned = best_log_reg_model.predict(X_test)

    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    report_tuned = classification_report(y_test, y_pred_tuned, zero_division=0)

    print(f"\nTuned Logistic Regression Model Performance on Test Set:")
    print(f"Accuracy: {accuracy_tuned:.4f}")
    print("\nClassification Report:")
    print(report_tuned)

    # --- 7. Save the Tuned Model and its associated TF-IDF Vectorizer ---
    print("\n--- Saving Tuned Model and Vectorizer ---")
    joblib.dump(best_log_reg_model, tuned_model_output_path)
    joblib.dump(tfidf_vectorizer, tuned_vectorizer_output_path) # Save the same vectorizer used for features
    print(f"Tuned Model saved to: {tuned_model_output_path}")
    print(f"Vectorizer used for tuned model saved to: {tuned_vectorizer_output_path}")

    print("\n--- Phase 5: Model Evaluation and Optimization (Hyperparameter Tuning) Completed ---")

except FileNotFoundError:
    print(f"Error: The input file '{file_path_input}' was not found. Please ensure '{file_path_input}' exists in your environment.")
except Exception as e:
    print(f"An unexpected error occurred during Hyperparameter Tuning: {e}")
