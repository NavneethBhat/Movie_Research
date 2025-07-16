Movie Review Sentiment Analysis Project
=======================================

Project Overview
----------------

This project focuses on building a machine learning model to perform sentiment analysis on movie reviews. Initially, the goal was to classify individual movie reviews into sentiment categories (Positive, Neutral, Negative). The project later expanded to analyze how these sentiments vary across different movie genres, providing valuable insights for filmmakers, producers, and marketers.

Problem Statement
-----------------

Streaming platforms and review aggregators host a diverse range of movie genres, each appealing to different audience segments and generating distinct viewer feedback. However, there is limited research on how user reviews and ratings vary across these genres. This study aims to analyze genre-based differences in movie reviews and ratings. By applying Natural Language Processing (NLP) techniques and statistical analysis, we identify key sentiment trends, common review themes, and rating patterns across different movie genres (e.g., Drama, Action, Comedy, Thriller). The findings will help filmmakers, producers, and marketers better understand genre-specific viewer expectations and inform future film production and promotional strategies.

Solution Architecture & Phases
------------------------------

The solution involves a typical machine learning pipeline, progressing through several key phases:

### Phase 1: Data Collection

*   **Process:** The project starts with a dataset of synthetic movie reviews, including review text, user ratings, and genre information.
    
*   **Reference File:** synthetic\_movie\_reviews\_from\_imdb.csv
    

### Phase 2: Data Preprocessing

*   **Process:** Raw review text undergoes several cleaning and normalization steps to prepare it for machine learning.
    
    *   **Text Cleaning:** Convert text to lowercase, remove URLs, special characters, numerical digits, and extra whitespace.
        
    *   **Tokenization & Filtering:** Split text into individual words (tokens), then remove common English stopwords.
        
    *   **Lemmatization:** Reduce words to their base or root form (e.g., "running" to "run") to reduce vocabulary size and improve consistency.
        
*   **Sentiment Label Derivation:** The User Rating column is used to create the target sentiment labels (Positive, Neutral, Negative) for model training:
    
    *   Ratings >= 7: 'Positive'
        
    *   Ratings <= 4: 'Negative'
        
    *   Ratings between 5 and 6: 'Neutral'
        

### Phase 3: Feature Engineering

*   **Process:** The cleaned and preprocessed text data is transformed into a numerical format that machine learning models can understand.
    
    *   **TF-IDF Vectorization:** Term Frequency-Inverse Document Frequency is used to convert text reviews into a matrix of numerical features. This technique reflects how important a word is to a document in a collection.
        

### Phase 4: Model Training

*   **Process:** A machine learning model is trained on the numerical features derived from the review text to predict the sentiment labels.
    
    *   **Algorithm:** Logistic Regression, a robust and interpretable algorithm suitable for multi-class text classification, is used.
        
    *   The model learns the relationship between word patterns and sentiment categories.
        

### Phase 5: Model Evaluation & Saving

*   **Process:** The trained model's performance is evaluated using metrics like accuracy and a classification report (precision, recall, F1-score) on a held-out test set.
    
    *   **Saving:** The trained Logistic Regression model and the TF-IDF Vectorizer are saved to disk. This allows for their reuse without retraining, which is crucial for deployment and future inference tasks.
        
*   **Output Files:**
    
    *   tuned\_logistic\_regression\_sentiment\_model.joblib
        
    *   tfidf\_vectorizer\_for\_tuned\_model.joblib
        

### Phase 6: Model Inference & Analysis

This phase involves using the trained model to make predictions and performing deeper analysis and visualization.

#### 6.1 Feature Importance Visualization

*   **Process:** The coefficients of the Logistic Regression model are analyzed to understand which words (features) have the most significant positive or negative impact on the prediction of each sentiment class.
    
*   **Purpose:** Provides interpretability to the model, showing "why" it classifies a review in a certain way. Helps in understanding domain-specific language associated with sentiments.
    
*   **Output Visualization:** logistic\_regression\_coefficients\_plot.png (a bar chart displaying top influential words per sentiment class).
    

#### 6.2 Genre-Based Sentiment Analysis & Visualization

*   **Process:** The trained sentiment model is applied to all reviews in the dataset, and the predicted sentiments are then aggregated and analyzed based on their respective Primary Genre.
    
*   **Purpose:** Directly addresses the expanded problem statement by identifying sentiment trends and patterns across different movie genres. This analysis helps filmmakers and marketers understand genre-specific viewer expectations.
    
*   **Output Text:** A detailed breakdown of sentiment proportions (Positive, Neutral, Negative) for each genre printed to the console.
    
*   **Output Visualization:** genre\_sentiment\_distribution.png (a stacked bar chart showing sentiment distribution per genre).
    

Key Deliverables & Outputs
--------------------------

Upon successful execution of the complete script, the following outputs are generated:

*   **Trained Model:** tuned\_logistic\_regression\_sentiment\_model.joblib
    
*   **Trained TF-IDF Vectorizer:** tfidf\_vectorizer\_for\_tuned\_model.joblib
    
*   **Feature Importance Plot:** logistic\_regression\_coefficients\_plot.png
    
*   **Genre Sentiment Distribution Plot:** genre\_sentiment\_distribution.png
    
*   **Console Output:** Model evaluation metrics (accuracy, classification report), sentiment distribution tables per genre, and key observations.
    

How to Run the Code
-------------------

1.  **Save the Script:** Save the provided Python script (containing all phases) to a .py file (e.g., sentiment\_analysis\_pipeline.py).
    
2.  **Place Data Files:** Ensure the synthetic\_movie\_reviews\_from\_imdb.csv file is in the **same directory** as your Python script.
    
3.  Bashpip install pandas scikit-learn nltk matplotlib seaborn joblib
    
4.  **Download NLTK Data:** The script will attempt to download necessary NLTK data (stopwords, wordnet) if not already present. Ensure you have an internet connection for this step.
    
5.  Bash python sentiment\_analysis\_pipeline.py
    

The script will print progress and results to the console, and generate the image files (.png) in the same directory.
<img width="1400" height="800" alt="genre_sentiment_distribution (1)" src="https://github.com/user-attachments/assets/76c28df5-ebdf-4e0c-944d-ca62d28dbe10" />
<img width="1800" height="1200" alt="logistic_regression_coefficients_plot (1)" src="https://github.com/user-attachments/assets/997ba1d9-4d50-43f5-aa9f-dc2dcd353eeb" />

