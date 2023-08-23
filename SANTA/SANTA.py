import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch import nn
from sklearn.metrics import confusion_matrix
from Data.data_utils import split_data
from Data.data_cleaning import preprocess_text, filter_data


class SANTA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # Add a channel dimension
        # embedded = [batch size, 1, sent len, emb dim]

        # Apply convolutions
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n]]

        # Apply max pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        # Concatenate pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


def train_model(data, vocab, UNK_ID):
    # Device agnostic code
    device = torch.device("cpu")

    # Identify keywords you want to pick up
    keywords = ["Inflation", "Interest rates", "Unemployment rates", "Retail sales", "Nonfarm payrolls", "CPI", "PPI",
                "Core inflation", "Stimulus", "Tapering", "FED"]

    # Filter the data based on keywords
    filtered_data = filter_data(data, keywords)

    # Apply preprocess_text function to title and description for filtered articles
    for article in filtered_data:
        article["title"] = preprocess_text(article["title"], vocab, UNK_ID)
        article["description"] = preprocess_text(article["description"], vocab, UNK_ID)

    # Preprocess and split data and move to device
    X_train, X_test, y_train, y_test = split_data(filtered_data)

    # Create an instance of the SANTA model
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 3
    DROPOUT = 0.5
    lr = 0.001

    model = SANTA(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 12
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        outputs = outputs.view(-1, OUTPUT_DIM)     # Reshape outputs to match batch size of labels
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted_labels = torch.max(outputs, 1)  # Get predicted labels
        correct_predictions = (predicted_labels == y_train).sum().item()
        accuracy = correct_predictions / y_train.size(0)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {(accuracy*100):.4f} %")

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get predictions
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = test_outputs.view(-1, OUTPUT_DIM)

    # Get predicted labels
    _, test_predicted_labels = torch.max(test_outputs, 1)

    # Calculate accuracy on the test data
    correct_predictions = (test_predicted_labels == y_test).sum().item()
    test_accuracy = correct_predictions / y_test.size(0) * 100

    print(f"Test Accuracy: {test_accuracy:.4f}%")

    # Get predicted labels from the test data
    test_predicted_labels = test_predicted_labels.cpu().numpy()  # Convert to numpy array

    # Create confusion matrix
    cm = confusion_matrix(y_test.cpu().numpy(), test_predicted_labels)

    # Calculate accuracy on the test data
    test_accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix\nTest Accuracy: {test_accuracy:.2f}%')
    plt.show()

    # Save the model to the specified path
    model_filename = "santa_model.pth"
    model_path = os.path.join("C:\\Coding\\Python\\SANTA", model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model
