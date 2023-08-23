# SANTA
Sentiment Analysis News Trading AI

# Description
A CNN is used to create a sentiment analysis model which provides signals to traders that trade USD. GNews API is used to take in data from Google news and return sentiment on USD.

# Table of Contents

1. Getting started
2. Usage
3. Features
4. Contributing
5. License

# Getting Started
Very easy to get started. Just clone this repo and run the main file. Feel free to make changes to suit your own needs.

# Usage
## Data collection
First of all, data needs to be collected to train the model. This is done by using the GNews API which collects articles in the Business section of the website. (It currently seems the max articles you can fetch on a free subscription is 3500 but feel free to test this further. The more articles gathered the better.)

Articles are then filtered based on a selection of keywords which are likely to affect USD. After this is done, articles are assigned a sentiment using `SentimentIntensityAnalyzer()`. 0 -> Neutral, 1 -> Buy, 2 -> Sell.

Next, data is turned into tensors and split into training and test datasets, ready to be entered into the model. A vocab list is also created at the same time.

## Santa Model Architecture
The SANTA model is designed for sentiment analysis on financial news data. It utilizes a convolutional neural network (CNN) architecture for this purpose. Here's an explanation of its key components:

### Model Initialization

```python
class SANTA(nn.Module):
def init(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
super().init()
```

- vocab_size: The size of the vocabulary, representing the number of unique words in your dataset.
- embedding_dim: The dimension of the word embeddings.
- n_filters: The number of filters (also known as kernels) in each convolutional layer.
- filter_sizes: A list of filter sizes for the convolutional layers.
- output_dim: The dimension of the model's output, typically representing sentiment classes (e.g., positive, negative, neutral).
- dropout: A dropout probability to prevent overfitting.

### Model Layers 

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim)
self.convs = nn.ModuleList([
nn.Conv2d(in_channels=1,
          out_channels=n_filters,
          kernel_size=(fs, embedding_dim))
          for fs in filter_sizes])
self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
self.dropout = nn.Dropout(dropout)
```

- nn.Embedding: This layer is responsible for converting word indices into dense word embeddings.
- nn.ModuleList: Multiple convolutional layers are created in parallel using this list comprehension. Each layer has a different kernel size defined by filter_sizes.
- nn.Linear: This fully connected layer performs the final classification based on the extracted features from convolutional layers.
- nn.Dropout: Dropout is applied to the concatenated features to reduce overfitting.

### Forward Pass

```python
def forward(self, text):
    embedded = self.embedding(text)
    embedded = embedded.unsqueeze(1)

    conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    cat = self.dropout(torch.cat(pooled, dim=1))

    return self.fc(cat)
```
- embedded: Converts input text data into word embeddings.
- conved: Applies convolutional operations with ReLU activation.
- pooled: Performs max pooling over the convolutional outputs to capture the most important features.
- cat: Concatenates the pooled features.

The final result is passed through a fully connected layer (self.fc) to produce sentiment predictions.

## Predictions

Finally, predictions are made by collecting raw data by again using the GNews API. Currently, business articles from the last 15 minutes are collected and then predictions are made on them.

Before predictions can be made, the same pre-processing method is applied to the new batch of articles and then fed into a saved version of the SANTA model. 

The articles headline, it's url and finally it's sentiment is returned to the user. The user may then use this information to help them place a trade.


# Features

- Sentiment analysis of financial news articles
- Real-time signal generation based on sentiment analysis
- Customizable signal thresholds

# Installation
The following libraries are required to run this code:

- PyTorch
- json
- os
- numpy
- matplotlib.pyplot
- seaborn
- sklearn
- gnews
- pickle
- pandas
- NLTK

# Acknowledgments
Thank you to the GNews API for massively increasing the ease of data collection. More info on Gnews can be found at:

https://github.com/ranahaani/GNews

# Contact Information
Please direct any questions to any of the following:

- e-mail: ronniepiku1@hotmail.co.uk

- LinkedIn: https://www.linkedin.com/in/ronald-piku/