import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter # Needed for Genre Analysis

# --- IMPORTANT: Make sure this file_path is correct for your VS Code environment ---
file_path = 'synthetic_movie_reviews_from_imdb.csv' # Or use '/full/path/to/synthetic_movie_reviews_from_imdb.csv'

try:
    df = pd.read_csv(file_path)

    # Ensure 'Sentiment' column exists (run this if it's a fresh load)
    def get_sentiment(rating):
        if rating >= 7:
            return 'Positive'
        elif rating >= 4:
            return 'Neutral'
        else:
            return 'Negative'
    df['Sentiment'] = df['User Rating'].apply(get_sentiment)
    print("DataFrame loaded and 'Sentiment' column ensured.")

    # --- 1. User Rating vs. Sentiment (Confirm Mapping) ---
    print("\n--- User Rating vs. Sentiment Sample ---")
    print(df[['User Rating', 'Sentiment']].sample(10))
    print("\n--- Sentiment Counts per User Rating ---")
    print(df.groupby('User Rating')['Sentiment'].value_counts().unstack(fill_value=0))

    # --- 2. Sentiment Distribution (Re-run for confirmation) ---
    print("\n--- Sentiment Distribution ---")
    sentiment_counts = df['Sentiment'].value_counts()
    print(sentiment_counts)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Distribution of Review Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.show()

    # --- 3. Genre Analysis ---
    print("\n--- Primary Genre Distribution (Top 15) ---")
    primary_genre_counts = df['Primary Genre'].value_counts().head(15)
    print(primary_genre_counts)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=primary_genre_counts.index, y=primary_genre_counts.values, palette='coolwarm')
    plt.title('Top 15 Primary Genre Distribution')
    plt.xlabel('Primary Genre')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\n--- All Genres Distribution (Top 20) ---")
    all_genres_list = df['All Genres'].str.split(', ').explode()
    all_genres_counts = Counter(all_genres_list)
    all_genres_series = pd.Series(all_genres_counts).sort_values(ascending=False).head(20)
    print(all_genres_series)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=all_genres_series.index, y=all_genres_series.values, palette='magma')
    plt.title('Top 20 Overall Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- 4. Review Length Analysis ---
    df['Review Length (Chars)'] = df['Review Text'].apply(len)
    df['Review Length (Words)'] = df['Review Text'].apply(lambda x: len(x.split()))
    print("\n--- Descriptive Statistics for Review Length (Characters) ---")
    print(df['Review Length (Chars)'].describe())
    print("\n--- Descriptive Statistics for Review Length (Words) ---")
    print(df['Review Length (Words)'].describe())
    plt.figure(figsize=(12, 6))
    sns.histplot(df, x='Review Length (Words)', hue='Sentiment', multiple='stack', bins=30, palette='viridis')
    plt.title('Distribution of Review Length (Words) by Sentiment')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # --- 5. Review Date Analysis ---
    df['Review Date'] = pd.to_datetime(df['Review Date'])
    df['Review Month'] = df['Review Date'].dt.to_period('M')
    reviews_per_month = df['Review Month'].value_counts().sort_index()
    print("\n--- Reviews per Month (Sample) ---")
    print(reviews_per_month.head())
    print(reviews_per_month.tail())
    plt.figure(figsize=(15, 7))
    reviews_per_month.plot(kind='line', marker='o', linestyle='-', color='skyblue')
    plt.title('Number of Reviews Over Time (Monthly)')
    plt.xlabel('Review Month')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- 6. Helpfulness Votes Analysis ---
    print("\n--- Descriptive Statistics for Helpfulness Votes ---")
    print(df['Helpfulness Votes'].describe())
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Helpfulness Votes'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Helpfulness Votes')
    plt.xlabel('Helpfulness Votes')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Sentiment', y='Helpfulness Votes', data=df, palette='pastel')
    plt.title('Helpfulness Votes by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Helpfulness Votes')
    plt.tight_layout()
    plt.show()
    correlation = df['User Rating'].corr(df['Helpfulness Votes'])
    print(f"\nCorrelation between User Rating and Helpfulness Votes: {correlation:.2f}")

    # --- 7. Source Platform Analysis ---
    print("\n--- Source Platform Distribution ---")
    source_platform_counts = df['Source Platform'].value_counts()
    print(source_platform_counts)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=source_platform_counts.index, y=source_platform_counts.values, palette='cubehelix')
    plt.title('Distribution of Reviews by Source Platform')
    plt.xlabel('Source Platform')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path in your VS Code environment.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
