import json
import os


# Function to build vocabulary from data
def build_vocab(data, min_word_freq=10, vocab_file="vocab.json", save_path="C:\\Coding\\Python\\SANTA"):
    vocab_file_path = os.path.join(save_path, vocab_file)

    if os.path.exists(vocab_file_path):
        # Load existing vocab file if it exists
        with open(vocab_file_path, "r") as f:
            vocab = json.load(f)
            return vocab, vocab["<UNK>"]

    word_counts = {}  # Create an empty dictionary to store word counts
    for article in data:  # Loop through each article in the data
        title_tokens = article["title"].split()  # Split title into tokens
        description_tokens = article["description"].split()  # Split description into tokens

        # Count the occurrences of each token in both title and description
        for token in title_tokens + description_tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    vocab = {"<PAD>": 0, "<UNK>": 1}  # Initialize vocabulary with padding and unknown tokens
    for token, count in word_counts.items():  # Loop through word counts
        if count >= min_word_freq:
            vocab[token] = len(vocab)  # Add token to vocabulary if count meets threshold

    # Save vocab to a file
    with open(vocab_file_path, "w") as f:
        json.dump(vocab, f)

    return vocab, vocab["<UNK>"]
