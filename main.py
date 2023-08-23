import os
import torch
import json
from SANTA.SANTA import SANTA, train_model
from Data.data_collector import get_articles
from Data.data_cleaning import preprocess_text
from Data.data_utils import get_data_with_sentiment

# Device agnostic code
device = torch.device("cpu")

# Load data
vocab_filename = "vocab.json"
vocab_path = os.path.join("C:\\Coding\\Python\\SANTA", vocab_filename)

model_filename = "santa_model.pth"
model_path = os.path.join("C:\\Coding\\Python\\SANTA", model_filename)

cache_filename = 'data_cache.pkl'
cached_path = os.path.join("C:\\Coding\\Python\\SANTA", cache_filename)

if os.path.exists(vocab_path) and os.path.exists(cached_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        UNK_ID = vocab["<UNK>"]
    data, _, _ = get_data_with_sentiment()
else:
    # Create a new vocabulary using get_data_with_sentiment
    data, vocab, UNK_ID = get_data_with_sentiment()
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)

# Fetch news articles using get_articles function
language = "en"
country = "US"
period = "0.25h"
max_results = 10
exclude_websites = None

new_articles = get_articles(language=language,
                            country=country,
                            period=period,
                            max_results=max_results,
                            exclude_websites=exclude_websites)

articles, title, description, url = new_articles

# Preprocess fetched articles
preprocessed_articles = []
for article in articles:
    title = preprocess_text(article["title"], vocab, UNK_ID)
    description = preprocess_text(article["description"], vocab, UNK_ID)
    preprocessed_articles.append({"title": title, "description": description})

# Load or train the model
if os.path.exists(model_path):
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 3
    DROPOUT = 0.5

    model = SANTA(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded pre-trained model.\n")
else:
    # Train a new model
    print("Training a new model...\n")
    model = train_model(data, vocab, UNK_ID)

# Process and make predictions for each fetched article
# Use zip to iterate over original and preprocessed articles
for article, preprocessed_article in zip(articles, preprocessed_articles):
    title_tensor = torch.LongTensor(preprocessed_article["title"]).unsqueeze(0).to(device)
    description_tensor = torch.LongTensor(preprocessed_article["description"]).unsqueeze(0).to(device)

    # Get the model's prediction
    with torch.no_grad():
        prediction = model(title_tensor)  # Pass both title and description tensors
        _, predicted_label = torch.max(prediction, 1)

    # Convert sentiment labels to Buy, Sell, Neutral
    sentiment_labels = {0: "Neutral", 1: "Buy", 2: "Sell"}
    predicted_sentiment = sentiment_labels[predicted_label.item()]

    # Print or store results
    print("Title:", article["title"])
    print("URL:", article["url"])
    print("Predicted Sentiment:", predicted_sentiment)
    print("------------------------------------\n")
