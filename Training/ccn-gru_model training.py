import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Custom Dataset for Pre-Extracted Features
class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)  # Shape: [20000, 171]
        self.labels = np.load(labels_path)  # Shape: [20000]
        # Reshape for CNN: [samples, channels=1, sequence_length=171]
        self.features = self.features.reshape(-1, 1, 171)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])


# CNN + GRU Model
class CNNGRUModel(nn.Module):
    def __init__(self, input_channels=1, cnn_filters=8, gru_hidden_dim=32, num_layers=2, num_classes=2, dropout=0.6):
        super(CNNGRUModel, self).__init__()
        # CNN layers (reduced capacity)
        self.conv1 = nn.Conv1d(input_channels, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        # GRU layer
        self.gru = nn.GRU(cnn_filters, gru_hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout)  # Fixed input_size
        # Fully connected layer
        self.fc = nn.Linear(gru_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN part
        x = self.pool(self.relu(self.conv1(x)))  # [batch, 8, 171//2 = 85]
        x = self.pool(self.relu(self.conv2(x)))  # [batch, 8, 85//2 = 42]
        x = x.permute(0, 2, 1)  # [batch, seq_len=42, features=8]

        # GRU part
        gru_out, _ = self.gru(x)  # Input: [batch, 42, 8]; Output: [batch, 42, gru_hidden_dim]
        x = gru_out[:, -1, :]  # Take the last time step: [batch, gru_hidden_dim]
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Training Function with Validation and Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Store predictions and labels for metrics
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Store predictions and labels for metrics
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        val_accuracy = accuracy_score(val_labels, val_preds)
        train_precision = precision_score(train_labels, train_preds)
        val_precision = precision_score(val_labels, val_preds)
        train_recall = recall_score(train_labels, train_preds)
        val_recall = recall_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds)
        val_f1 = f1_score(val_labels, val_preds)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, '
              f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')

        # Early stopping and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


# Main Execution
if __name__ == "__main__":
    # Define paths
    base_dir = "C:/Users/91826/PycharmProjects/fakeaudiodetection"
    features_dir = os.path.join(base_dir, "processed_features")
    save_dir = os.path.join(base_dir, "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    train_features_path = os.path.join(features_dir, "1111111train_test_features.npy")
    train_labels_path = os.path.join(features_dir, "1111111train_test_labels.npy")
    model_save_path = os.path.join(save_dir, "mixed_audio_cnn_gru.pth")

    # Verify files exist
    if not os.path.exists(train_features_path) or not os.path.exists(train_labels_path):
        print(f"Error: Feature or label file not found at {train_features_path} or {train_labels_path}")
        exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and split into train and validation
    full_dataset = FeatureDataset(train_features_path, train_labels_path)
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Initialize model, criterion, and optimizer
    model = CNNGRUModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate to limit accuracy

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device,
                save_path=model_save_path)
    print(f"Model training completed and saved at {model_save_path}")