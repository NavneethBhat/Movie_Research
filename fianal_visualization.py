import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration for file paths ---
original_data_path = 'synthetic_movie_reviews_from_imdb.csv' # This should contain the 'genre' column
model_input_path = 'tuned_logistic_regression_sentiment_model.joblib'
vectorizer_input_path = 'tfidf_vectorizer_for_tuned_model.joblib'
output_plot_genre_sentiment = 'genre_sentiment_distribution.png'

# --- NLTK Downloads Check (essential for preprocessing) ---
# Ensure these are downloaded if you run the script in a new environment
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

# --- Text Preprocessing Functions (consistent with previous phases) ---
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

print("\n--- Genre-Based Sentiment Analysis and Visualization Script ---")
print("--- IMPORTANT: Ensure 'nltk' library is installed (pip install nltk). ---")
print("--- Also confirm that 'synthetic_movie_reviews_from_imdb.csv', 'tuned_logistic_regression_sentiment_model.joblib', and 'tfidf_vectorizer_for_tuned_model.joblib' are in the SAME DIRECTORY as this script. ---")

# The main logic is enclosed in a try-except block to handle potential errors
try:
    # --- 1. Load the Original Dataset with Genre Information ---
    print(f"\nAttempting to load original dataset from: {original_data_path}")
    df = pd.read_csv(original_data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Basic check for essential columns with CORRECTED NAMES
    if 'Review Text' not in df.columns or 'Primary Genre' not in df.columns:
        raise ValueError("The dataset must contain 'Review Text' and 'Primary Genre' columns. Please check your CSV file.")
    
    # --- 2. Preprocess Review Text ---
    print("\nPreprocessing review text for sentiment prediction...")
    # Use 'Review Text' column for preprocessing
    df['processed_review'] = df['Review Text'].apply(lambda x: preprocess_text(clean_text(x)))
    print("Review text preprocessing complete.")

    # --- 3. Load the Saved Model and TF-IDF Vectorizer ---
    print("\nAttempting to load Tuned Model and Vectorizer...")
    loaded_model = joblib.load(model_input_path)
    loaded_vectorizer = joblib.load(vectorizer_input_path)
    print(f"Model loaded from: {model_input_path}")
    print(f"Vectorizer loaded from: {vectorizer_input_path}")

    # --- 4. Transform Processed Reviews using the Loaded Vectorizer ---
    print("\nTransforming processed reviews into TF-IDF features...")
    tfidf_features = loaded_vectorizer.transform(df['processed_review'])
    print(f"TF-IDF Feature Matrix created with shape: {tfidf_features.shape}")

    # --- 5. Predict Sentiment for All Reviews ---
    print("\nPredicting sentiment for all reviews...")
    df['predicted_sentiment'] = loaded_model.predict(tfidf_features)
    print("Sentiment prediction complete.")

    # --- 6. Analyze Genre-Based Sentiment Trends (Text Output) ---
    print("\n--- Analyzing Genre-Based Sentiment Trends ---")
    
    # Group by 'Primary Genre' (CORRECTED) and predicted sentiment, then count occurrences
    genre_sentiment_counts = df.groupby(['Primary Genre', 'predicted_sentiment']).size().unstack(fill_value=0)
    
    # Normalize counts to get proportions for each sentiment within each genre
    genre_sentiment_proportions = genre_sentiment_counts.apply(lambda x: x / x.sum(), axis=1)
    
    print("\nSentiment Distribution (Proportions) Across Genres:")
    # Using .to_string() for better console readability of the full table
    print(genre_sentiment_proportions.to_string())

    print("\n--- Key Observations from Genre-Based Sentiment ---")
    # Identify and print the genres with the highest/lowest proportions for each sentiment
    # Only if the sentiment column exists (it should, after unstack and fill_value=0)
    if 'Positive' in genre_sentiment_proportions.columns:
        print(f"Most Positive Genre(s): {genre_sentiment_proportions['Positive'].idxmax()} ({genre_sentiment_proportions['Positive'].max():.2f})")
        print(f"Least Positive Genre(s): {genre_sentiment_proportions['Positive'].idxmin()} ({genre_sentiment_proportions['Positive'].min():.2f})")
    if 'Negative' in genre_sentiment_proportions.columns:
        print(f"Most Negative Genre(s): {genre_sentiment_proportions['Negative'].idxmax()} ({genre_sentiment_proportions['Negative'].max():.2f})")
        print(f"Least Negative Genre(s): {genre_sentiment_proportions['Negative'].idxmin()} ({genre_sentiment_proportions['Negative'].min():.2f})")
    if 'Neutral' in genre_sentiment_proportions.columns:
        print(f"Most Neutral Genre(s): {genre_sentiment_proportions['Neutral'].idxmax()} ({genre_sentiment_proportions['Neutral'].max():.2f})")
        print(f"Least Neutral Genre(s): {genre_sentiment_proportions['Neutral'].idxmin()} ({genre_sentiment_proportions['Neutral'].min():.2f})")


    # --- 7. Genre-Based Sentiment Visualization ---
    print("\n--- Generating Genre-Based Sentiment Visualization ---")
    
    # Define a consistent order for sentiment categories for plotting
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    
    # Ensure all sentiment columns exist in the proportions DataFrame, filling with 0 if absent
    for col in sentiment_order:
        if col not in genre_sentiment_proportions.columns:
            genre_sentiment_proportions[col] = 0.0
    
    # Reindex the DataFrame to ensure the desired plotting order for sentiments
    genre_sentiment_proportions = genre_sentiment_proportions[sentiment_order]

    # Create a stacked bar plot using pandas built-in plotting (which uses matplotlib)
    # colormap='viridis' provides a good range of distinct colors
    genre_sentiment_proportions.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
    
    # Set plot title and labels
    plt.title('Sentiment Distribution Across Movie Genres', fontsize=16)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Proportion of Reviews', fontsize=12)
    
    # Rotate x-axis labels (genres) for better readability if many genres
    plt.xticks(rotation=45, ha='right') 
    
    # Move the legend outside the plot to prevent obscuring bars
    plt.legend(title='Predicted Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left') 
    
    # Adjust plot layout to prevent labels/titles from overlapping
    plt.tight_layout() 
    
    # Save the generated plot to an image file
    plt.savefig(output_plot_genre_sentiment)
    print(f"Genre sentiment distribution plot saved to: {output_plot_genre_sentiment}")

    print("\n--- Genre-Based Sentiment Analysis and Visualization Script Completed ---")

# --- Error Handling ---
except FileNotFoundError as e:
    print(f"\nError: A required file was not found. Please ensure all input files (e.g., {original_data_path}, {model_input_path}, {vectorizer_input_path}) exist in the same directory as your script. Details: {e}")
except ValueError as e:
    print(f"\nData Error: {e}. Please check your dataset's columns ('Review Text', 'Primary Genre') and content.")
except ModuleNotFoundError as e:
    print(f"\nModule Not Found Error: {e}. Please install the missing library (e.g., 'pip install nltk' for NLTK, 'pip install seaborn' for Seaborn, 'pip install pandas' for Pandas, 'pip install matplotlib' for Matplotlib).")
except Exception as e:
    print(f"\nAn unexpected error occurred during execution: {e}")
