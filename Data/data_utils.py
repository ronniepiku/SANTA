import pickle
import os
import pandas as pd
import torch
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from Data.data_collector import collect_data
from Data.data_vocab_utils import build_vocab

# Device agnostic code
device = torch.device("cpu")


# Function to save data to cache
def save_to_cache(data, cache_filename):
    cache_path = os.path.join("C:\\Coding\\Python\\SANTA", cache_filename)
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(data, cache_file)


# Function to load data from cache
def load_from_cache(cache_filename):
    cache_path = os.path.join("C:\\Coding\\Python\\SANTA", cache_filename)
    try:
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    except FileNotFoundError:
        return None


# Function to get data with sentiment and perform preprocessing
def get_data_with_sentiment():

    # Download 'vader_lexicon' if it doesn't exist, and skip if it does
    nltk.download('vader_lexicon', quiet=True)

    cache_filename = 'data_cache.pkl'

    # Try to load cached data
    cached_data = load_from_cache(cache_filename)

    if cached_data is not None:
        print("Using cached data.")
        data = cached_data[0]
        vocab = cached_data[1]
        unk_id = cached_data[2]
    else:
        print("Fetching new data...")
        data = collect_data()  # Fetch new data to compare
        vocab, unk_id = build_vocab(data)  # Build vocab and get UNK_ID
        save_to_cache((data, vocab, unk_id), cache_filename)  # Save collected data, vocab, and UNK_ID to cache

    print("Total number of fetched articles:", len(data))

    # Perform sentiment analysis and assign labels
    sia = SentimentIntensityAnalyzer()
    for article in data:
        title_sentiment = sia.polarity_scores(article["title"])
        description_sentiment = sia.polarity_scores(article["description"])

        title_compound = title_sentiment["compound"]
        description_compound = description_sentiment["compound"]

        if title_compound > 0.1 and description_compound > 0.1:
            article["sentiment"] = 1
        elif title_compound < -0.1 and description_compound < -0.1:
            article["sentiment"] = 2
        else:
            article["sentiment"] = 0

    return data, vocab, unk_id


def split_data(filtered_data):
    # Preprocessed text data (tokenized and preprocessed)
    X_title = [article["title"] for article in filtered_data]  # List of tokenized titles
    X_description = [article["description"] for article in filtered_data]  # List of tokenized descriptions

    # Labels
    y = [article["sentiment"] for article in filtered_data]  # List of labels

    # Determine the maximum length of titles and descriptions
    max_title_length = max(len(title) for title in X_title)
    max_description_length = max(len(description) for description in X_description)

    # Pad titles and descriptions to the maximum lengths
    X_title_padded = [title + [0] * (max_title_length - len(title)) for title in X_title]
    X_description_padded = [description + [0] * (max_description_length - len(description)) for description in
                            X_description]

    # Convert to NumPy arrays
    X_title_array = np.array(X_title_padded)
    X_description_array = np.array(X_description_padded)
    X = np.column_stack((X_title_array, X_description_array))
    y_array = np.array(y)

    # Turn data into tensors
    X = torch.from_numpy(X).type(torch.long)
    y = torch.from_numpy(y_array).type(torch.long)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    return X_train, X_test, y_train, y_test


# Function to create a DataFrame from post-processed data
def create_df(post_processed_data):
    data = []

    for article in post_processed_data:
        data.append([article["title"], article["description"], article["published date"],
                     article["url"], article["publisher"]["title"], article["sentiment"]])

    # Create a pandas DataFrame with specified columns
    columns = ["Title", "Description", "Published Date", "URL", "Publisher", "Sentiment"]
    df = pd.DataFrame(data, columns=columns)

    return df
