import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
# No longer need requests or time for API calls
# import requests
# import time

# --- Configuration for IMDb Datasets ---
# IMPORTANT: You must upload these files to your Google Colab environment
# or ensure they are in the same directory if running locally.
IMDB_BASICS_FILE = "title.basics.tsv.gz"
IMDB_RATINGS_FILE = "title.ratings.tsv.gz"

def load_movies_from_imdb_tsv(basics_file, ratings_file, min_votes=1000):
    """
    Loads and processes movie data from IMDb's title.basics.tsv.gz and title.ratings.tsv.gz.

    Args:
        basics_file (str): Path to the title.basics.tsv.gz file.
        ratings_file (str): Path to the title.ratings.tsv.gz file.
        min_votes (int): Minimum number of votes for a movie to be included.

    Returns:
        list: A list of dictionaries, each containing movie title, primary_genre, and all_genres.
    """
    print(f"Loading IMDb basics data from {basics_file}...")
    try:
        # Use low_memory=False to avoid mixed type warnings for large files
        basics_df = pd.read_csv(basics_file, sep='\t', low_memory=False, na_values=['\\N'])
        print(f"Loaded {len(basics_df)} records from basics.")
    except FileNotFoundError:
        print(f"Error: {basics_file} not found. Please ensure it's uploaded or in the correct path.")
        return []
    except Exception as e:
        print(f"Error reading {basics_file}: {e}")
        return []

    print(f"Loading IMDb ratings data from {ratings_file}...")
    try:
        ratings_df = pd.read_csv(ratings_file, sep='\t', na_values=['\\N'])
        print(f"Loaded {len(ratings_df)} records from ratings.")
    except FileNotFoundError:
        print(f"Error: {ratings_file} not found. Please ensure it's uploaded or in the correct path.")
        return []
    except Exception as e:
        print(f"Error reading {ratings_file}: {e}")
        return []

    print("Merging basics and ratings data...")
    # Merge the two dataframes on 'tconst' (IMDb ID)
    merged_df = pd.merge(basics_df, ratings_df, on='tconst', how='inner')
    print(f"Merged dataframe has {len(merged_df)} records.")

    # Filter for movies and shorts (excluding TV series, video games, etc.)
    # IMDb uses 'movie' and 'short' for typical film entries.
    movie_types = ['movie', 'short']
    filtered_df = merged_df[merged_df['titleType'].isin(movie_types)].copy()

    # Filter out adult content
    filtered_df = filtered_df[filtered_df['isAdult'] == 0].copy()

    # Filter out entries with missing titles, genres, or insufficient votes
    filtered_df = filtered_df.dropna(subset=['primaryTitle', 'genres', 'averageRating', 'numVotes']).copy()
    filtered_df = filtered_df[filtered_df['numVotes'] >= min_votes].copy()

    print(f"Filtered to {len(filtered_df)} movie titles after cleaning.")

    movies_list = []
    for index, row in filtered_df.iterrows():
        title = row['primaryTitle']
        genres_str = str(row['genres']) # Ensure it's a string

        # Split genres and clean them
        all_genres = [g.strip() for g in genres_str.split(',') if g.strip()]

        # Determine primary genre (take the first, or default if list is empty)
        primary_genre = all_genres[0] if all_genres else "Unknown"

        if primary_genre != "Unknown": # Only include if we have a valid genre
            movies_list.append({
                "title": title,
                "primary_genre": primary_genre,
                "all_genres": all_genres
            })

    print(f"Prepared {len(movies_list)} movies for review generation.")
    return movies_list

def generate_movie_review_dataset(num_records=100, imdb_basics_file=None, imdb_ratings_file=None):
    """
    Generates a synthetic dataset of movie reviews, fetching movies from IMDb TSV files.

    Args:
        num_records (int): The number of synthetic records to generate.
        imdb_basics_file (str): Path to the title.basics.tsv.gz file.
        imdb_ratings_file (str): Path to the title.ratings.tsv.gz file.

    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic movie review data.
    """

    movies_data = []
    if imdb_basics_file and imdb_ratings_file:
        movies_data = load_movies_from_imdb_tsv(imdb_basics_file, imdb_ratings_file)
        if not movies_data:
            print("Failed to load movies from IMDb TSV files. Falling back to a static list.")

    # Fallback to a small static list if loading from TSV fails
    if not movies_data:
        movies_data = [
            {"title": "The Shawshank Redemption", "primary_genre": "Drama", "all_genres": ["Drama"]},
            {"title": "The Dark Knight", "primary_genre": "Action", "all_genres": ["Action", "Crime", "Drama", "Thriller"]},
            {"title": "Pulp Fiction", "primary_genre": "Crime", "all_genres": ["Crime", "Drama"]},
            {"title": "Forrest Gump", "primary_genre": "Drama", "all_genres": ["Drama", "Romance"]},
            {"title": "Inception", "primary_genre": "Science Fiction", "all_genres": ["Action", "Adventure", "Science Fiction", "Thriller"]},
            {"title": "The Matrix", "primary_genre": "Science Fiction", "all_genres": ["Action", "Science Fiction"]},
            {"title": "Parasite", "primary_genre": "Thriller", "all_genres": ["Comedy", "Drama", "Thriller"]},
            {"title": "Whiplash", "primary_genre": "Drama", "all_genres": ["Drama", "Music"]},
            {"title": "Bridesmaids", "primary_genre": "Comedy", "all_genres": ["Comedy", "Romance"]},
            {"title": "Godzilla Minus One", "primary_genre": "Action", "all_genres": ["Action", "Science Fiction"]},
            {"title": "Dune: Part Two", "primary_genre": "Science Fiction", "all_genres": ["Science Fiction", "Adventure"]},
            {"title": "Oppenheimer", "primary_genre": "Drama", "all_genres": ["Drama", "History"]},
        ]
        print(f"Using a static fallback list of {len(movies_data)} movies for generation.")
    else:
        print(f"Using {len(movies_data)} movies loaded from IMDb TSV files for generation.")

    # --- Review Text Samples ---
    review_texts_positive = [
        "Absolutely loved it! A masterpiece of storytelling and emotion. Highly recommend.",
        "Fantastic movie! The acting was superb and the plot kept me engaged from start to finish.",
        "Brilliant! Every scene was captivating and the ending was perfect. A truly immersive experience.",
        "A must-watch! Laughed/cried/was thrilled throughout. Exceeded all my expectations.",
        "One of the best movies I've seen in years. Truly unforgettable and thought-provoking.",
        "So well-crafted and visually stunning. An instant classic.",
        "The performances were stellar, bringing depth and nuance to every character.",
        "Phenomenal! A cinematic triumph in every sense, with innovative direction.",
        "Completely blown away! The narrative depth and intricate details were astounding.",
        "A joyous experience from start to finish. Pure entertainment and emotional resonance.",
    ]

    review_texts_neutral = [
        "It was okay. Had some good moments but also some slow parts, making it uneven.",
        "Decent watch, but nothing particularly groundbreaking or memorable.",
        "An average film. Enjoyable enough for a casual viewing, but didn't stand out.",
        "Not bad, not great. Pretty standard for its genre, with predictable elements.",
        "Had its ups and downs. Some parts were stronger than others, creating inconsistencies.",
        "A passable film. Didn't leave a strong impression, either way, neither good nor bad.",
        "Competent, but lacked that extra spark to make it truly compelling or original.",
        "Felt a bit generic at times, but still held my attention adequately.",
        "Some interesting ideas, but the execution was inconsistent, leading to missed potential.",
        "An alright movie. Wouldn't rush to rewatch it, but it served its purpose.",
    ]

    review_texts_negative = [
        "Disappointing. The plot was weak and the acting unconvincing, making it hard to watch.",
        "Not good at all. Found it boring and poorly directed, with numerous pacing issues.",
        "A complete waste of time. Avoid at all costs, as it offers little entertainment.",
        "Terrible! Lacked any redeeming qualities, feeling disjointed and poorly conceived.",
        "Could not get into it. Overly long and confusing, with a muddled storyline.",
        "Felt like a chore to finish. Very uninspired and derivative, lacking originality.",
        "Poorly written and executed. A big letdown from the director/franchise.",
        "Seriously flawed. Too many plot holes and bad decisions in character development.",
        "Wouldn't recommend it. Left me feeling unsatisfied and wishing I had chosen another film.",
        "A real mess from start to finish. Big thumbs down for its lack of cohesion.",
    ]

    all_review_texts = {
        "positive": review_texts_positive,
        "neutral": review_texts_neutral,
        "negative": review_texts_negative
    }

    data = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 7, 15) # Current date

    if not movies_data:
        print("Warning: No movie data available. Cannot generate reviews.")
        return pd.DataFrame() # Return empty DataFrame if no movies

    for i in range(num_records):
        movie = random.choice(movies_data)

        if 'title' not in movie or 'primary_genre' not in movie or 'all_genres' not in movie:
            continue

        movie_id = f"M{random.randint(1000, 9999)}"
        review_id = f"R{i + 1:05d}"
        reviewer_id = f"U{random.randint(10000, 99999)}"

        rating = random.randint(1, 10)
        helpfulness_votes = random.randint(0, 750)

        if rating >= 7:
            sentiment_category = "positive"
        elif rating >= 4:
            sentiment_category = "neutral"
        else:
            sentiment_category = "negative"

        review_text = random.choice(all_review_texts[sentiment_category])

        if random.random() < 0.35:
            extra_text_pool = all_review_texts["positive"] + all_review_texts["neutral"] + all_review_texts["negative"]
            review_text += " " + random.choice(extra_text_pool)
            if random.random() < 0.15:
                review_text += " " + random.choice(extra_text_pool)

        random_days = random.randint(0, (end_date - start_date).days)
        review_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

        source_platform = random.choice(["IMDb", "Rotten Tomatoes", "Metacritic", "Letterboxd"])

        data.append({
            "Review ID": review_id,
            "Movie ID": movie_id,
            "Movie Title": movie["title"],
            "Primary Genre": movie["primary_genre"],
            "All Genres": ", ".join(movie["all_genres"]),
            "User Rating": rating,
            "Review Text": review_text,
            "Review Date": review_date,
            "Reviewer ID": reviewer_id,
            "Helpfulness Votes": helpfulness_votes,
            "Source Platform": source_platform,
        })

    df = pd.DataFrame(data)
    return df

# --- How to use the script ---
if __name__ == "__main__":
    # In Google Colab, you'll need to upload 'title.basics.tsv.gz' and 'title.ratings.tsv.gz'
    # to the session storage.
    # You can do this by clicking the 'Files' icon on the left sidebar
    # (folder icon), then the 'Upload to session storage' icon (page with an arrow up).

    # Ensure pandas is installed (if running locally or in a fresh Colab environment):
    # !pip install pandas

    # Generate 25,000 records. This will reuse the loaded movies as needed.
    synthetic_df = generate_movie_review_dataset(
        num_records=25000, # Changed from 10000 to 25000
        imdb_basics_file=IMDB_BASICS_FILE,
        imdb_ratings_file=IMDB_RATINGS_FILE
    )

    if not synthetic_df.empty:
        print("\nGenerated Synthetic Dataset Head:")
        print(synthetic_df.head())

        print("\nDataset Info:")
        synthetic_df.info()

        output_filename = "synthetic_movie_reviews_from_imdb.csv"
        synthetic_df.to_csv(output_filename, index=False)
        print(f"\nDataset saved to {output_filename}")

        print("\nPrimary Genre Distribution (from loaded IMDb movies):")
        print(synthetic_df["Primary Genre"].value_counts().head(15)) # Show top 15 genres
    else:
        print("No data generated. Please ensure 'title.basics.tsv.gz' and 'title.ratings.tsv.gz' are correctly uploaded and accessible.")
