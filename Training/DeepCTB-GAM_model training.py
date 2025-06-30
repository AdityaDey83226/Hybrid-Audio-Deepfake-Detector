import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom Dataset for Pre-Extracted Features
class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)  # Shape: [20000, 171]
        self.labels = np.load(labels_path)  # Shape: [20000]
        # Reshape for CNN: [samples, channels, sequence_length]
        self.features = self.features.reshape(-1, 1, 171)  # [20000, 1, 171]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

# Global Attention Mechanism
class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        # hidden_size * 2 because BiLSTM outputs are bidirectional
        self.attention = nn.Linear(hidden_size * 2, 1)  # Additive attention scoring
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_size * 2]
        # Compute attention scores
        attention_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = self.softmax(attention_scores)  # [batch, seq_len, 1]
        # Compute weighted sum of lstm_out
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_size * 2]
        return context_vector, attention_weights

# CNN-BiLSTM-Transformer Model with Global Attention
class CNNBiLSTMTransformerModel(nn.Module):
    def __init__(self, input_size=171, hidden_size=64, num_layers=2, num_classes=2, dropout=0.5):
        super(CNNBiLSTMTransformerModel, self).__init__()
        # CNN Layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32,  # Matches conv2 output channels
                nhead=4,  # Number of attention heads (must divide d_model)
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        # Global Attention Layer
        self.attention = GlobalAttention(hidden_size)
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # CNN processing
        x = self.pool(self.relu(self.conv1(x)))  # [batch, 16, 85]
        x = self.pool(self.relu(self.conv2(x)))  # [batch, 32, 42]
        # Reshape for Transformer: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, 42, 32]
        # Transformer processing
        x = self.transformer_encoder(x)  # [batch, 42, 32]
        # BiLSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, 42, hidden_size * 2]
        # Global Attention
        context_vector, attention_weights = self.attention(lstm_out)  # [batch, hidden_size * 2]
        # Apply dropout and fully connected layer
        x = self.dropout(context_vector)
        x = self.fc(x)  # [batch, num_classes]
        return x

# Training Function with Metrics and Warmup
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=30, warmup_epochs=5):
    model.train()
    model.to(device)
    base_lr = 0.0001
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Store predictions and labels for metrics
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics for the epoch
        epoch_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        # Print metrics
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1-Score: {f1:.4f}')

# Main Execution
if __name__ == "__main__":
    # Define paths
    base_dir = "C:/Users/91826/PycharmProjects/fakeaudiodetection"
    features_dir = os.path.join(base_dir, "processed_features")
    save_dir = os.path.join(base_dir, "saved_model")
    train_features_path = os.path.join(features_dir, "1111111train_test_features.npy")
    train_labels_path = os.path.join(features_dir, "1111111train_test_labels.npy")
    model_save_path = os.path.join(save_dir, "mixed_audio_cnn_bilstm_transformer_att.pth")

    # Verify files exist
    if not os.path.exists(train_features_path) or not os.path.exists(train_labels_path):
        print(f"Error: Feature or label file not found at {train_features_path} or {train_labels_path}")
        exit(1)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create dataset and dataloader
    train_dataset = FeatureDataset(train_features_path, train_labels_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMTransformerModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully at {model_save_path}")