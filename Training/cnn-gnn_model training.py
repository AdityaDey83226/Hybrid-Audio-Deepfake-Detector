import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Custom Dataset for Pre-Extracted Features (Converted to Graphs after CNN)
class GraphFeatureDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)  # Shape: [20000, 171]
        self.labels = np.load(labels_path)      # Shape: [20000]
        # Reshape for CNN: [samples, channels=1, sequence_length=171]
        self.features = self.features.reshape(-1, 1, 171)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

# CNN-GNN Hybrid Model
class CNNGNNModel(nn.Module):
    def __init__(self, input_channels=1, cnn_filters=8, gnn_hidden_dim=64, num_classes=2, dropout=0.3):
        super(CNNGNNModel, self).__init__()
        # CNN Part (same as CNN-LSTM)
        self.conv1 = nn.Conv1d(input_channels, cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        # GNN Part
        self.conv1_gnn = GCNConv(cnn_filters, gnn_hidden_dim)
        self.conv2_gnn = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.fc = nn.Linear(gnn_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_size):
        # CNN processing
        x = self.pool(self.relu(self.conv1(x)))  # [batch, 8, 171//2 = 85]
        x = self.pool(self.relu(self.conv2(x)))  # [batch, 8, 85//2 = 42]
        # Prepare for GNN: Treat each of the 42 time steps as a node with 8 features
        x = x.permute(0, 2, 1)  # [batch, 42, 8]
        num_nodes_per_graph = x.shape[1]  # 42 nodes per graph
        x = x.reshape(-1, x.shape[2])  # [batch * 42, 8]

        # Create edge indices for a linear graph (0-1, 1-2, ..., 40-41) for each sample
        edge_index = []
        for b in range(batch_size):
            start_node = b * num_nodes_per_graph
            for i in range(num_nodes_per_graph - 1):
                edge_index.append([start_node + i, start_node + i + 1])
                edge_index.append([start_node + i + 1, start_node + i])  # Bidirectional
        edge_index = torch.LongTensor(edge_index).t().to(x.device)  # [2, num_edges]

        # Create batch tensor for global pooling
        batch = torch.repeat_interleave(torch.arange(batch_size, device=x.device), num_nodes_per_graph)

        # GNN processing
        x = self.conv1_gnn(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2_gnn(x, edge_index)
        x = self.relu(x)
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)  # [batch, gnn_hidden_dim]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, num_classes]
        return x

# Training Function with Validation and Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            batch_size = inputs.shape[0]
            optimizer.zero_grad()
            outputs = model(inputs, batch_size)
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
                batch_size = inputs.shape[0]
                outputs = model(inputs, batch_size)
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
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
    model_save_path = os.path.join(save_dir, "mixed_audio_cnn_gnn.pth")

    # Verify files exist
    if not os.path.exists(train_features_path) or not os.path.exists(train_labels_path):
        print(f"Error: Feature or label file not found at {train_features_path} or {train_labels_path}")
        exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and split into train and validation
    full_dataset = GraphFeatureDataset(train_features_path, train_labels_path)
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Initialize model, criterion, and optimizer
    model = CNNGNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device, save_path=model_save_path)
    print(f"Model training completed and saved at {model_save_path}")