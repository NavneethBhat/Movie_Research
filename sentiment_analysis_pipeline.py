import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration for file paths ---
original_data_path = 'synthetic_movie_reviews_from_imdb.csv'
trained_model_output_path = 'tuned_logistic_regression_sentiment_model.joblib'
trained_vectorizer_output_path = 'tfidf_vectorizer_for_tuned_model.joblib'
output_plot_feature_importance = 'logistic_regression_coefficients_plot.png'
output_plot_genre_sentiment = 'genre_sentiment_distribution.png'

# --- NLTK Downloads Check and Text Preprocessing Functions ---
print("Checking NLTK data downloads...")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    print("NLTK resources already available.")
except LookupError:
    print("Downloading NLTK resources (stopwords, wordnet, omw-1.4)...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK resources downloaded successfully!")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmas)

print("\n--- Starting All Phases of Sentiment Analysis Project ---")
print("--- Ensure all required CSV and .joblib files are in the same directory as this script. ---")
print("--- Ensure all Python libraries (nltk, pandas, scikit-learn, joblib, matplotlib, seaborn) are installed. ---")

try:
    # --- PHASE 1 & 2: Data Collection & Initial Preprocessing Setup (Functions Defined Above) ---
    # --- PHASE 3: Preprocessing Execution ---
    # --- PHASE 4: Feature Engineering (TF-IDF) ---

    print("\n--- Phase: Data Loading, Preprocessing, and Feature Engineering ---")
    df = pd.read_csv(original_data_path)
    print(f"Original dataset loaded. Shape: {df.shape}")

    # Check for required columns and preprocess 'Review Text'
    if 'Review Text' not in df.columns or 'User Rating' not in df.columns or 'Primary Genre' not in df.columns:
        raise ValueError("Dataset must contain 'Review Text', 'User Rating', and 'Primary Genre' columns.")

    df['cleaned_review'] = df['Review Text'].apply(clean_text)
    df['processed_review'] = df['cleaned_review'].apply(preprocess_text)
    print("Review text cleaned and preprocessed.")

    # Convert User Rating to Sentiment (Example Logic, adjust as needed based on your actual rating scale)
    # Assuming User Rating is 1-10
    def get_sentiment(rating):
        if rating >= 7:
            return 'Positive'
        elif rating <= 4:
            return 'Negative'
        else:
            return 'Neutral'
            
    df['sentiment'] = df['User Rating'].apply(get_sentiment)
    print("User Ratings converted to Sentiment categories (Positive, Neutral, Negative).")

    # Split data for training
    X = df['processed_review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Initialize and fit TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting features for manageability
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print(f"TF-IDF Vectorizer fitted. Number of features: {X_train_tfidf.shape[1]}")

    # --- PHASE 5: Model Training (Logistic Regression) ---
    print("\n--- Phase: Model Training (Logistic Regression) ---")
    model = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
    model.fit(X_train_tfidf, y_train)
    print("Logistic Regression model trained.")

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- PHASE 6: Model Saving & Loading ---
    print("\n--- Phase: Model Saving and Loading ---")
    joblib.dump(model, trained_model_output_path)
    joblib.dump(tfidf_vectorizer, trained_vectorizer_output_path)
    print(f"Trained model saved to: {trained_model_output_path}")
    print(f"TF-IDF Vectorizer saved to: {trained_vectorizer_output_path}")

    # Load the saved model and vectorizer for subsequent tasks
    loaded_model = joblib.load(trained_model_output_path)
    loaded_vectorizer = joblib.load(trained_vectorizer_output_path)
    print("Model and Vectorizer re-loaded successfully for further use.")

    # --- Feature Importance Visualization ---
    print("\n--- Phase: Feature Importance Visualization (Logistic Regression Coefficients) ---")
    
    feature_names = loaded_vectorizer.get_feature_names_out()
    coefficients = loaded_model.coef_
    classes = loaded_model.classes_

    n_top_features = 15
    plt.figure(figsize=(18, 12))
    plt.suptitle('Top {} Influential Words (Features) for Each Sentiment Class'.format(n_top_features), fontsize=16)

    for i, class_label in enumerate(classes):
        class_coefficients = coefficients[i]
        coef_series = pd.Series(class_coefficients, index=feature_names)
        top_positive_coefs = coef_series.nlargest(n_top_features)
        top_negative_coefs = coef_series.nsmallest(n_top_features)
        combined_coefs = pd.concat([top_negative_coefs, top_positive_coefs]).sort_values()

        ax = plt.subplot(1, len(classes), i + 1)
        colors = ['red' if c < 0 else 'green' for c in combined_coefs]
        ax.barh(combined_coefs.index, combined_coefs.values, color=colors)
        ax.set_title(f'Class: {class_label}')
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Word (Feature)')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_plot_feature_importance)
    print(f"Feature importance visualization saved to: {output_plot_feature_importance}")

    # --- Genre-Based Sentiment Analysis and Visualization ---
    print("\n--- Phase: Genre-Based Sentiment Analysis and Visualization ---")

    # Apply preprocessing and prediction to the full original DataFrame
    df['processed_review_for_genre_analysis'] = df['Review Text'].apply(lambda x: preprocess_text(clean_text(x)))
    tfidf_features_all = loaded_vectorizer.transform(df['processed_review_for_genre_analysis'])
    df['predicted_sentiment_genre'] = loaded_model.predict(tfidf_features_all)
    print("Sentiments predicted for all reviews based on genre analysis.")

    # Calculate sentiment distribution per genre
    genre_sentiment_counts = df.groupby(['Primary Genre', 'predicted_sentiment_genre']).size().unstack(fill_value=0)
    genre_sentiment_proportions = genre_sentiment_counts.apply(lambda x: x / x.sum(), axis=1)
    
    print("\nSentiment Distribution (Proportions) Across Genres:")
    print(genre_sentiment_proportions.to_string())

    print("\n--- Key Observations from Genre-Based Sentiment ---")
    if 'Positive' in genre_sentiment_proportions.columns:
        print(f"Most Positive Genre(s): {genre_sentiment_proportions['Positive'].idxmax()} ({genre_sentiment_proportions['Positive'].max():.2f})")
        print(f"Least Positive Genre(s): {genre_sentiment_proportions['Positive'].idxmin()} ({genre_sentiment_proportions['Positive'].min():.2f})")
    if 'Negative' in genre_sentiment_proportions.columns:
        print(f"Most Negative Genre(s): {genre_sentiment_proportions['Negative'].idxmax()} ({genre_sentiment_proportions['Negative'].max():.2f})")
        print(f"Least Negative Genre(s): {genre_sentiment_proportions['Negative'].idxmin()} ({genre_sentiment_proportions['Negative'].min():.2f})")
    if 'Neutral' in genre_sentiment_proportions.columns:
        print(f"Most Neutral Genre(s): {genre_sentiment_proportions['Neutral'].idxmax()} ({genre_sentiment_proportions['Neutral'].max():.2f})")
        print(f"Least Neutral Genre(s): {genre_sentiment_proportions['Neutral'].idxmin()} ({genre_sentiment_proportions['Neutral'].min():.2f})")

    # Plotting Genre-Based Sentiment
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    for col in sentiment_order:
        if col not in genre_sentiment_proportions.columns:
            genre_sentiment_proportions[col] = 0.0
    genre_sentiment_proportions = genre_sentiment_proportions[sentiment_order]

    genre_sentiment_proportions.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
    plt.title('Sentiment Distribution Across Movie Genres', fontsize=16)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Proportion of Reviews', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Predicted Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_plot_genre_sentiment)
    print(f"Genre sentiment distribution plot saved to: {output_plot_genre_sentiment}")

    print("\n--- All Phases of Sentiment Analysis Project Completed Successfully ---")

except FileNotFoundError as e:
    print(f"\nError: A required file was not found. Please ensure all input files (CSV, .joblib) exist in the same directory as your script. Details: {e}")
except ValueError as e:
    print(f"\nData Error: {e}. Please check your dataset's columns and content.")
except ModuleNotFoundError as e:
    print(f"\nModule Not Found Error: {e}. Please install the missing library (e.g., 'pip install nltk', 'pip install seaborn', etc.).")
except Exception as e:
    print(f"\nAn unexpected error occurred during execution: {e}")
