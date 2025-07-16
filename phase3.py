import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- NLTK Data Downloads (Corrected Error Handling) ---
# These are necessary for stopwords and lemmatization.
# This block will now correctly check and download if resources are missing.
print("Checking NLTK data downloads...")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4') # Open Multilingual Wordnet
    print("NLTK resources already available.")
except LookupError: # Corrected: Catch LookupError when data is not found
    print("Downloading NLTK resources...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK resources downloaded successfully!")
# --- End NLTK Data Downloads ---

# --- Configuration for file paths ---
# Input file from Phase 2 (EDA processed data)
file_path_input = 'synthetic_movie_reviews_eda_processed.csv'
# Output file for Phase 3 (text processed data)
file_path_output = 'synthetic_movie_reviews_text_processed.csv'

try:
    # --- 1. Loading the Processed Data (from Phase 2) ---
    df = pd.read_csv(file_path_input)
    print(f"\nDataFrame loaded from {file_path_input}")
    print("Original 'Review Text' sample:")
    print(df['Review Text'].head())

    # Ensure 'Sentiment' column exists (if loading from a CSV that didn't include it, though it should from EDA phase)
    if 'Sentiment' not in df.columns:
        def get_sentiment(rating):
            if rating >= 7:
                return 'Positive'
            elif rating >= 4:
                return 'Neutral'
            else:
                return 'Negative'
        df['Sentiment'] = df['User Rating'].apply(get_sentiment)
        print("Note: 'Sentiment' column was re-created as it was not found in the loaded DataFrame.")
    else:
        print("Note: 'Sentiment' column already exists in the loaded DataFrame.")


    # --- 2. Text Cleaning Function ---
    def clean_text(text):
        text = text.lower() # Lowercasing
        text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
        text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
        text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation, keep words and spaces
        text = re.sub(r'\n', '', text) # Remove newline characters
        text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
        return text

    df['cleaned_review'] = df['Review Text'].apply(clean_text)
    print("\n--- 'cleaned_review' sample (after cleaning): ---")
    print(df['cleaned_review'].head())

    # --- 3. Tokenization and 4. Stop Word Removal & 5. Lemmatization ---
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = text.split() # Simple tokenization by splitting on space
        filtered_tokens = [word for word in tokens if word not in stop_words] # Remove stop words
        lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens] # Lemmatization
        return ' '.join(lemmas)

    df['processed_review'] = df['cleaned_review'].apply(preprocess_text)
    print("\n--- 'processed_review' sample (after tokenization, stop word removal, lemmatization): ---")
    print(df['processed_review'].head())

    # Display some statistics or confirmation
    print(f"\nShape of DataFrame after text preprocessing: {df.shape}")
    print(f"New columns added: {['cleaned_review', 'processed_review']}")

    # --- Save the DataFrame with processed text ---
    df.to_csv(file_path_output, index=False)
    print(f"\nDataFrame with processed text saved to {file_path_output}")

except FileNotFoundError:
    print(f"Error: The input file '{file_path_input}' was not found. Please ensure '{file_path_input}' exists in your VS Code environment.")
except Exception as e:
    print(f"An error occurred during text preprocessing: {e}")
