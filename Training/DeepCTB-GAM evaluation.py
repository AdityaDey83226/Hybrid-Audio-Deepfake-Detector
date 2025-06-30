import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import psutil

# Custom Dataset for Pre-Extracted Features
class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path, mapping_path=None):
        self.features = np.load(features_path)  # Shape: [num_samples, 171]
        self.labels = np.load(labels_path)  # Shape: [num_samples]
        self.mapping = np.load(mapping_path, allow_pickle=True) if mapping_path else None  # Shape: [num_samples, 2]
        # Reshape for CNN: [samples, channels, sequence_length]
        self.features = self.features.reshape(-1, 1, 171)

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
                nhead=4,  # Number of attention heads
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

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Evaluation Function with Maximum Metrics, Losses (Segment and File-Level), Confusion Matrix, ROC Curve, and Scalability Metrics
def evaluate_model(model, test_loader, mapping, device, save_dir):
    # Ensure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC and ROC curve
    file_preds = {}
    file_true = {}
    file_probs = {}  # For file-level metrics
    ce_losses = []
    bce_losses = []
    focal_losses = []

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')  # Cross-Entropy Loss (also Log Loss in this context)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')  # BCEWithLogitsLoss for raw logits
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')  # Focal Loss
    total_samples = 0

    # Track inference time and memory usage
    start_time = time.time()
    process = psutil.Process()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Fix: Remove extra dimension, shape [batch_size, 1] -> [batch_size]
            batch_size = inputs.shape[0]
            total_samples += batch_size
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1 (Fake)
            _, preds = torch.max(outputs, 1)
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            all_probs.extend(probabilities)

            # Compute segment-level losses
            ce_loss = ce_loss_fn(outputs, labels)
            ce_losses.append(ce_loss.item())

            # For BCEWithLogitsLoss, convert labels to float and use raw logits for class 1
            bce_labels = labels.float()
            bce_logits = outputs[:, 1]  # Logits for class 1 (Fake)
            bce_loss = bce_loss_fn(bce_logits, bce_labels)
            bce_losses.append(bce_loss.item())

            # Focal Loss
            focal_loss = focal_loss_fn(outputs, labels)
            focal_losses.append(focal_loss.item())

            # Map predictions to files for majority voting
            batch_start = batch_idx * test_loader.batch_size
            batch_end = batch_start + len(batch_labels)
            batch_mapping = mapping[batch_start:batch_end]
            for i, (pred, label, prob) in enumerate(zip(batch_preds, batch_labels, probabilities)):
                file_name, _ = batch_mapping[i]
                if file_name not in file_preds:
                    file_preds[file_name] = []
                    file_true[file_name] = []
                    file_probs[file_name] = []
                file_preds[file_name].append(pred)
                file_true[file_name].append(label)
                file_probs[file_name].append(prob)

    # Calculate inference time and memory usage
    inference_time = time.time() - start_time
    peak_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB

    print(f"\n=== Scalability Metrics for 1000 Unseen Files ===")
    print(f"Processed {len(all_preds)} segments")
    print(f"Number of files: {len(file_preds)}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Inference Time per File: {inference_time / len(file_preds):.4f} seconds")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")

    # Compute average segment-level losses
    avg_ce_loss = sum(ce_losses) / total_samples
    avg_bce_loss = sum(bce_losses) / total_samples
    avg_focal_loss = sum(focal_losses) / total_samples

    # Segment-level metrics
    print("\n=== Segment-Level Metrics ===")
    print(f"Segment-level Cross-Entropy Loss (Log Loss): {avg_ce_loss:.4f}")
    print(f"Segment-level BCEWithLogits Loss: {avg_bce_loss:.4f}")
    print(f"Segment-level Focal Loss: {avg_focal_loss:.4f}")

    segment_accuracy = accuracy_score(all_labels, all_preds)
    segment_precision = precision_score(all_labels, all_preds, average='weighted')
    segment_recall = recall_score(all_labels, all_preds, average='weighted')
    segment_f1 = f1_score(all_labels, all_preds, average='weighted')
    segment_auc = roc_auc_score(all_labels, all_probs)
    segment_kappa = cohen_kappa_score(all_labels, all_preds)
    cm_segment = confusion_matrix(all_labels, all_preds)

    print(f'Segment-Level Accuracy: {segment_accuracy:.4f}')
    print(f'Segment-Level Precision: {segment_precision:.4f}')
    print(f'Segment-Level Recall: {segment_recall:.4f}')
    print(f'Segment-Level F1-Score: {segment_f1:.4f}')
    print(f'Segment-Level AUC: {segment_auc:.4f}')
    print(f'Segment-Level Cohen\'s Kappa: {segment_kappa:.4f}')
    print("\nSegment-Level Confusion Matrix:")
    print("[[True Negatives, False Positives]")
    print(" [False Negatives, True Positives]]")
    print(cm_segment)

    # Visualize segment-level confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_segment, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Segment-Level Confusion Matrix (1000 Unseen Files)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    segment_cm_path = os.path.join(save_dir, 'segment_confusion_matrix_1000.png')
    plt.savefig(segment_cm_path)
    plt.close()
    print(f"Saved Segment-Level Confusion Matrix image to: {segment_cm_path}")

    # Segment-level ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Segment-Level ROC Curve (1000 Unseen Files)')
    plt.legend(loc="lower right")
    plt.grid(True)
    segment_roc_path = os.path.join(save_dir, 'segment_roc_curve_1000.png')
    plt.savefig(segment_roc_path)
    plt.close()
    print(f"Saved Segment-Level ROC Curve image to: {segment_roc_path}")

    # Majority voting per file
    file_labels = []
    file_preds_final = []
    file_probs_final = []
    for file_name in file_preds:
        true_votes = np.mean(file_true[file_name])
        pred_votes = np.mean(file_preds[file_name])
        prob_votes = np.mean(file_probs[file_name])
        file_labels.append(1 if true_votes >= 0.5 else 0)
        file_preds_final.append(1 if pred_votes >= 0.5 else 0)
        file_probs_final.append(prob_votes)

    # Compute file-level losses
    file_labels_tensor = torch.LongTensor(file_labels).to(device)
    file_probs_tensor = torch.FloatTensor([[1-prob, prob] for prob in file_probs_final]).to(device)  # [num_files, 2] for CE/Focal Loss
    file_logits_tensor = torch.log(file_probs_tensor + 1e-10)  # Convert probs to logits for BCEWithLogitsLoss (add small epsilon to avoid log(0))

    file_ce_loss = ce_loss_fn(file_probs_tensor, file_labels_tensor).item() / len(file_labels)
    file_bce_loss = bce_loss_fn(file_logits_tensor[:, 1], file_labels_tensor.float()).item() / len(file_labels)
    file_focal_loss = focal_loss_fn(file_probs_tensor, file_labels_tensor).item() / len(file_labels)

    # File-level metrics
    print("\n=== File-Level Metrics ===")
    print(f"File-level Cross-Entropy Loss (Log Loss): {file_ce_loss:.4f}")
    print(f"File-level BCEWithLogits Loss: {file_bce_loss:.4f}")
    print(f"File-level Focal Loss: {file_focal_loss:.4f}")

    file_accuracy = accuracy_score(file_labels, file_preds_final)
    file_precision = precision_score(file_labels, file_preds_final, average='weighted')
    file_recall = recall_score(file_labels, file_preds_final, average='weighted')
    file_f1 = f1_score(file_labels, file_preds_final, average='weighted')
    file_auc = roc_auc_score(file_labels, file_probs_final)
    file_kappa = cohen_kappa_score(file_labels, file_preds_final)
    cm_file = confusion_matrix(file_labels, file_preds_final)

    print(f'Model Accuracy: {file_accuracy:.4f}')
    print(f'Model Precision: {file_precision:.4f}')
    print(f'Model Recall: {file_recall:.4f}')
    print(f'Model F1-Score: {file_f1:.4f}')
    print(f'Model AUC: {file_auc:.4f}')
    print(f'Model Cohen\'s Kappa: {file_kappa:.4f}')
    print("\nModel Confusion Matrix:")
    print("[[True Negatives, False Positives]")
    print(" [False Negatives, True Positives]]")
    print(cm_file)

    # Visualize file-level confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_file, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('File-Level Confusion Matrix (1000 Unseen Files)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    file_cm_path = os.path.join(save_dir, 'file_confusion_matrix_1000.png')
    plt.savefig(file_cm_path)
    plt.close()
    print(f"Saved File-Level Confusion Matrix image to: {file_cm_path}")

    # File-level ROC curve
    fpr_file, tpr_file, _ = roc_curve(file_labels, file_probs_final)
    roc_auc_file = auc(fpr_file, tpr_file)
    plt.figure()
    plt.plot(fpr_file, tpr_file, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_file:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('File-Level ROC Curve (1000 Unseen Files)')
    plt.legend(loc="lower right")
    plt.grid(True)
    file_roc_path = os.path.join(save_dir, 'file_roc_curve_1000.png')
    plt.savefig(file_roc_path)
    plt.close()
    print(f"Saved File-Level ROC Curve image to: {file_roc_path}")

    return (segment_accuracy, segment_precision, segment_recall, segment_f1, segment_auc, segment_kappa, cm_segment,
            file_accuracy, file_precision, file_recall, file_f1, file_auc, file_kappa, cm_file,
            inference_time, peak_memory)

# Main Execution
if __name__ == "__main__":
    # Define paths
    base_dir = "C:/Users/91826/PycharmProjects/fakeaudiodetection"
    features_dir = os.path.join(base_dir, "processed_features")
    save_dir = os.path.join(base_dir, "saved_model")
    test_features_path = os.path.join(features_dir, "1111111unseen_features.npy")
    test_labels_path = os.path.join(features_dir, "1111111unseen_labels.npy")
    test_mapping_path = os.path.join(features_dir, "1111111unseen_file_mapping.npy")
    model_path = os.path.join(save_dir, "mixed_audio_cnn_bilstm_transformer_att.pth")

    # Verify files exist
    if not os.path.exists(test_features_path):
        print(f"Error: Test features file not found at {test_features_path}.")
        exit(1)
    if not os.path.exists(test_labels_path):
        print(f"Error: Test labels file not found at {test_labels_path}.")
        exit(1)
    if not os.path.exists(test_mapping_path):
        print(f"Error: Test mapping file not found at {test_mapping_path}.")
        exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run the training script first.")
        exit(1)

    # Create dataset and dataloader
    test_dataset = FeatureDataset(test_features_path, test_labels_path, test_mapping_path)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMTransformerModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate the model with file-level majority voting and scalability metrics
    (segment_accuracy, segment_precision, segment_recall, segment_f1, segment_auc, segment_kappa, cm_segment,
     file_accuracy, file_precision, file_recall, file_f1, file_auc, file_kappa, cm_file,
     inference_time, peak_memory) = evaluate_model(model, test_loader, np.load(test_mapping_path, allow_pickle=True), device, save_dir)

    # Print a summary of key metrics
    print("\n=== Evaluation Summary for 1000 Unseen Files ===")
    print(f"Segment Accuracy: {segment_accuracy:.4f}")
    print(f"Segment AUC: {segment_auc:.4f}")
    print(f"File Accuracy: {file_accuracy:.4f}")
    print(f"File AUC: {file_auc:.4f}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Inference Time per File: {inference_time / 1000:.4f} seconds")  # 1000 files
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")