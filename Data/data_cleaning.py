import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Function to load cached data from a file
def load_cached_data(cache_filename):
    try:
        with open(cache_filename, 'rb') as cache_file:
            return pickle.load(cache_file)
    except FileNotFoundError:
        return None


# Function to filter data based on keywords
def filter_data(pre_processed_data, keywords):
    cleaned_data = []
    for article in pre_processed_data:
        # Check if any keyword is present in the lowercase title or description
        if any(keyword.lower() in article["title"].lower() or keyword.lower()
               in article["description"].lower() for keyword in keywords):
            cleaned_article = {
                "title": article["title"],
                "description": article["description"],
                "published date": article["published date"],
                "url": article["url"],
                "publisher": article["publisher"],
                "sentiment": article["sentiment"]
            }
            cleaned_data.append(cleaned_article)

    print("Total number of filtered articles:", len(cleaned_data))

    return cleaned_data


# Function to preprocess text using tokenization, lemmatization, and stopword removal
def preprocess_text(text, vocab, unk_id):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Map tokens to their integer IDs using the provided vocabulary
    ids = [vocab.get(token, unk_id) for token in tokens]
    return ids
