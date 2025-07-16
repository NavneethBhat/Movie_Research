import pandas as pd
import joblib # For loading models
import matplotlib.pyplot as plt # For plotting
import numpy as np # For numerical operations, though not directly used for plotting here

# --- Configuration for file paths ---
model_input_path = 'tuned_logistic_regression_sentiment_model.joblib'
vectorizer_input_path = 'tfidf_vectorizer_for_tuned_model.joblib'
output_plot_path = 'logistic_regression_coefficients_plot.png' # Where the plot image will be saved

# --- CRITICAL: FINAL CHECK FOR FILE PATHS ---
# If you are absolutely certain the files are in the same directory as your script,
# and this error persists, there might be an issue with how VS Code's Python environment
# is resolving the current working directory.
# You might try providing the FULL, ABSOLUTE PATH to the .joblib files if the problem continues.
print("\n--- ATTEMPTING VISUALIZATION AGAIN ---")
print("--- Please ensure 'tuned_logistic_regression_sentiment_model.joblib' and 'tfidf_vectorizer_for_tuned_model.joblib' are in the EXACT SAME DIRECTORY as this script. ---")
print(f"Attempting to load model from: {model_input_path}")
print(f"Attempting to load vectorizer from: {vectorizer_input_path}")
print("If FileNotFoundError occurs again, the files are NOT accessible at these paths from Python's perspective.")

try:
    print("\n--- Generating Model Visualization (Feature Importance) ---")

    # --- 1. Load the Saved Model and TF-IDF Vectorizer ---
    print("\n--- Loading Saved Model and Vectorizer ---")
    loaded_model = joblib.load(model_input_path)
    loaded_vectorizer = joblib.load(vectorizer_input_path)
    print(f"Model loaded from: {model_input_path}")
    print(f"Vectorizer loaded from: {vectorizer_input_path}")

    # --- 2. Extract Feature Names (Words) from the Vectorizer ---
    # These are the words that TF-IDF identified and vectorized
    feature_names = loaded_vectorizer.get_feature_names_out()
    print(f"\nExtracted {len(feature_names)} feature names from the vectorizer.")

    # --- 3. Extract Coefficients from the Loaded Model ---
    # For a multi-class Logistic Regression model (like ours with Positive/Neutral/Negative),
    # loaded_model.coef_ will be a 2D array. Each row corresponds to a class, and columns
    # correspond to the features (words).
    # loaded_model.classes_ gives the order of the classes (e.g., ['Negative', 'Neutral', 'Positive'])
    coefficients = loaded_model.coef_
    classes = loaded_model.classes_
    print(f"Extracted coefficients for {len(classes)} classes: {classes}")

    # --- 4. Visualize Top N Coefficients for each class ---
    n_top_features = 15 # Number of top positive and top negative features to display for each class

    plt.figure(figsize=(18, 12)) # Create a figure to hold multiple subplots for each class
    # Set a main title for the entire figure
    plt.suptitle('Top {} Influential Words (Features) for Each Sentiment Class (Logistic Regression Coefficients)'.format(n_top_features), fontsize=16)

    # Iterate through each class to create a separate subplot
    for i, class_label in enumerate(classes):
        # Get the array of coefficients specifically for the current class
        class_coefficients = coefficients[i]

        # Create a Pandas Series to easily associate coefficients with their feature names
        coef_series = pd.Series(class_coefficients, index=feature_names)

        # Get the top N words with the highest positive coefficients for this class
        top_positive_coefs = coef_series.nlargest(n_top_features)
        # Get the top N words with the lowest (most negative) coefficients for this class
        top_negative_coefs = coef_series.nsmallest(n_top_features)

        # Concatenate positive and negative coefficients and sort them for plotting
        # This arranges them from most negative to most positive
        combined_coefs = pd.concat([top_negative_coefs, top_positive_coefs]).sort_values()

        # Create a subplot within the figure (1 row, number of classes columns, current subplot index)
        ax = plt.subplot(1, len(classes), i + 1)
        
        # Assign colors based on coefficient sign (red for negative, green for positive)
        colors = ['red' if c < 0 else 'green' for c in combined_coefs]
        
        # Create the horizontal bar chart
        ax.barh(combined_coefs.index, combined_coefs.values, color=colors)
        
        # Set subplot title and labels
        ax.set_title(f'Class: {class_label}')
        ax.set_xlabel('Coefficient Value (Impact on Prediction)')
        ax.set_ylabel('Word (Feature)')
        
        # Adjust y-axis tick label size for better readability if many features
        ax.tick_params(axis='y', labelsize=10) 
        
        # Add a grid for better readability of values
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Automatically adjust subplot parameters for a tight layout,
        # leaving space for the main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    # Save the entire figure to a file
    plt.savefig(output_plot_path)
    print(f"\nCoefficient visualization saved to: {output_plot_path}")

    print("\n--- Model Visualization Completed ---")

except FileNotFoundError as e:
    print(f"Error: A required file was not found. Please ensure all input files ({model_input_path}, {vectorizer_input_path}) exist in your environment. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred during visualization: {e}")
