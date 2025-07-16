import pandas as pd
import joblib # For loading models
import re # For text cleaning
import nltk # For stop words and lemmatization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Ensure NLTK data is downloaded (if you're running this script in a new environment
# without having run Phase 3's full script that includes the downloads)
print("Checking NLTK data downloads...")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    print("NLTK resources already available.")
except LookupError: # CORRECTED: Catch LookupError here
    print("Downloading NLTK resources (stopwords, wordnet, omw-1.4)...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK resources downloaded successfully!")

# --- Configuration for file paths ---
model_input_path = 'tuned_logistic_regression_sentiment_model.joblib'
vectorizer_input_path = 'tfidf_vectorizer_for_tuned_model.joblib'
predictions_output_path = 'predicted_movie_review_sentiments.csv' # New: Path to save predictions

try:
    print("--- Starting Phase 6: Inference/Prediction ---")

    # --- 1. Load the Saved Model and TF-IDF Vectorizer ---
    print("\n--- Loading Saved Model and Vectorizer ---")
    loaded_model = joblib.load(model_input_path)
    loaded_vectorizer = joblib.load(vectorizer_input_path)
    print(f"Model loaded from: {model_input_path}")
    print(f"Vectorizer loaded from: {vectorizer_input_path}")

    # --- 2. Define New Movie Review Texts for Prediction ---
    new_reviews = [
        "This movie was absolutely brilliant! The acting was superb and the story captivated me from start to finish. A must-watch!",
        "It was okay. Not great, not terrible. A bit bland, but watchable if you have nothing else to do.",
        "An absolute disaster. The plot made no sense, the characters were annoying, and I wasted two hours of my life. Avoid at all costs.",
        "The cinematography was stunning, but the pacing dragged considerably. A visual feast, but narratively weak.",
        "Decent action sequences, but the dialogue felt forced. Enjoyable enough for a Friday night."
    ]
    print("\n--- New Review Texts for Prediction ---")
    for i, review in enumerate(new_reviews):
        print(f"Review {i+1}: {review}")

    # --- 3. Apply the Same Text Preprocessing Steps from Phase 3 ---
    # Define cleaning and preprocessing functions again to ensure consistency
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

    processed_new_reviews = [preprocess_text(clean_text(review)) for review in new_reviews]

    print("\n--- Processed New Review Texts ---")
    for i, processed_review in enumerate(processed_new_reviews):
        print(f"Processed Review {i+1}: {processed_review}")

    # --- 4. Transform New Texts into TF-IDF Features using the LOADED Vectorizer ---
    # This is CRUCIAL. Do not fit_transform again! Only transform.
    new_reviews_tfidf = loaded_vectorizer.transform(processed_new_reviews)
    print(f"\nTransformed new reviews into TF-IDF features. Shape: {new_reviews_tfidf.shape}")

    # --- 5. Make Predictions using the LOADED Model ---
    predictions = loaded_model.predict(new_reviews_tfidf)
    print("\n--- Predictions ---")

    # --- 6. Display Results ---
    results_df = pd.DataFrame({
        'Original Review': new_reviews,
        'Processed Review': processed_new_reviews,
        'Predicted Sentiment': predictions
    })
    print(results_df.to_string()) # .to_string() for better display of full content

    # --- 7. NEW: Save Predictions to CSV ---
    results_df.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions saved to: {predictions_output_path}")

    print("\n--- Phase 6: Inference/Prediction Completed ---")

except FileNotFoundError as e:
    print(f"Error: A required file was not found. Please ensure all input files ({model_input_path}, {vectorizer_input_path}) exist in your environment. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred during Phase 6: {e}")
