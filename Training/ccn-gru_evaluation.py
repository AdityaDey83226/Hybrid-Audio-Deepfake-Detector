import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Custom Dataset for Pre-Extracted Features
class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path, mapping_path):
        self.features = np.load(features_path)  # Shape: [5000, 171]
        self.labels = np.load(labels_path)  # Shape: [5000]
        self.mapping = np.load(mapping_path, allow_pickle=True)  # Shape: [5000, 2]
        # Reshape for CNN: [samples, channels=1, sequence_length=171]
        self.features = self.features.reshape(-1, 1, 171)
        print(f"Features shape: {self.features.shape}, Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

# CNN + GRU Model (Matching the training script's architecture)
class CNNGRUModel(nn.Module):
    def __init__(self, input_channels=1, cnn_filters=8, gru_hidden_dim=32, num_layers=2, num_classes=2, dropout=0.6):
        super(CNNGRUModel, self).__init__()
        # CNN layers
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

# Evaluation Function with Losses (Segment and File-Level), Majority Voting, AUC, and ROC Curve
def evaluate_model(model, test_loader, mapping, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC and ROC calculation
    file_preds = {}
    file_true = {}
    file_probs = {}  # For file-level AUC and ROC
    ce_losses = []
    bce_losses = []
    focal_losses = []

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')  # Cross-Entropy Loss (also Log Loss in this context)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')  # BCEWithLogitsLoss for raw logits
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')  # Focal Loss
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
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

            # Map predictions to files
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

    print(f"Processed {len(all_preds)} segments")
    print(f"Number of files: {len(file_preds)}")

    # Compute average segment-level losses
    avg_ce_loss = sum(ce_losses) / total_samples
    avg_bce_loss = sum(bce_losses) / total_samples
    avg_focal_loss = sum(focal_losses) / total_samples
    print(f"Segment-level Cross-Entropy Loss (Log Loss): {avg_ce_loss:.4f}")
    print(f"Segment-level BCEWithLogits Loss: {avg_bce_loss:.4f}")
    print(f"Segment-level Focal Loss: {avg_focal_loss:.4f}")

    # Check consistency of segment labels within each file
    inconsistent_files = 0
    for file_name in file_true:
        labels = file_true[file_name]
        if len(set(labels)) > 1:  # If more than one unique label in the file
            inconsistent_files += 1
            print(f"File {file_name} has inconsistent segment labels: {labels}")
    print(f"Number of files with inconsistent segment labels: {inconsistent_files}")

    # Segment-level metrics (before majority voting)
    segment_cm = confusion_matrix(all_labels, all_preds)
    print(f"Segment-level Confusion Matrix:\n{segment_cm}")
    false_positives = segment_cm[0, 1]  # FP: True label 0, Predicted 1
    false_negatives = segment_cm[1, 0]  # FN: True label 1, Predicted 0
    misclassified_segments = false_positives + false_negatives
    segment_accuracy = (segment_cm[0, 0] + segment_cm[1, 1]) / np.sum(segment_cm)
    segment_auc = roc_auc_score(all_labels, all_probs)  # Segment-level AUC
    print(f"Segment-level Accuracy: {segment_accuracy:.4f}")
    print(f"Segment-level AUC: {segment_auc:.4f}")
    print(f"Misclassified Segments (FP + FN): {misclassified_segments} (FP: {false_positives}, FN: {false_negatives})")

    # Compute segment-level ROC curve
    segment_fpr, segment_tpr, _ = roc_curve(all_labels, all_probs)

    # Majority voting per file
    file_labels = []
    file_preds_final = []
    file_probs_final = []
    for file_name in file_preds:
        true_votes = np.mean(file_true[file_name])  # Average label (0 or 1)
        pred_votes = np.mean(file_preds[file_name])  # Average prediction
        prob_votes = np.mean(file_probs[file_name])  # Average probability for AUC
        file_labels.append(1 if true_votes >= 0.5 else 0)  # Majority true label
        file_preds_final.append(1 if pred_votes >= 0.5 else 0)  # Majority prediction
        file_probs_final.append(prob_votes)

    # Compute file-level losses
    file_labels_tensor = torch.LongTensor(file_labels).to(device)
    file_probs_tensor = torch.FloatTensor([[1-prob, prob] for prob in file_probs_final]).to(device)  # [num_files, 2] for CE/Focal Loss
    file_logits_tensor = torch.log(file_probs_tensor + 1e-10)  # Convert probs to logits for BCEWithLogitsLoss (add small epsilon to avoid log(0))

    file_ce_loss = ce_loss_fn(file_probs_tensor, file_labels_tensor).item() / len(file_labels)
    file_bce_loss = bce_loss_fn(file_logits_tensor[:, 1], file_labels_tensor.float()).item() / len(file_labels)
    file_focal_loss = focal_loss_fn(file_probs_tensor, file_labels_tensor).item() / len(file_labels)

    print(f"File-level Cross-Entropy Loss (Log Loss): {file_ce_loss:.4f}")
    print(f"File-level BCEWithLogits Loss: {file_bce_loss:.4f}")
    print(f"File-level Focal Loss: {file_focal_loss:.4f}")

    # Calculate file-level metrics
    accuracy = accuracy_score(file_labels, file_preds_final)
    precision = precision_score(file_labels, file_preds_final)
    recall = recall_score(file_labels, file_preds_final)
    f1 = f1_score(file_labels, file_preds_final)
    kappa = cohen_kappa_score(file_labels, file_preds_final)
    auc_score = roc_auc_score(file_labels, file_probs_final)  # File-level AUC
    cm = confusion_matrix(file_labels, file_preds_final)

    # Compute file-level ROC curve
    file_fpr, file_tpr, _ = roc_curve(file_labels, file_probs_final)

    # Print file-level metrics
    print(f'File-level Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Kappa Score: {kappa:.4f}')
    print(f'AUC Score: {auc_score:.4f}')
    print(f'File-level Confusion Matrix:\n{cm}')

    # Visualize file-level confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (File-Level)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cnn_gru_confusion_matrix.png')
    plt.close()

    # Visualize ROC curves (Segment-level and File-level)
    plt.figure(figsize=(8, 6))
    plt.plot(segment_fpr, segment_tpr, label=f'Segment-level ROC (AUC = {segment_auc:.4f})', color='blue')
    plt.plot(file_fpr, file_tpr, label=f'File-level ROC (AUC = {auc_score:.4f})', color='green')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for CNN-GRU Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('cnn_gru_roc_curve.png')
    plt.close()

    return accuracy, precision, recall, f1, kappa, auc_score, cm

# Main Execution
if __name__ == "__main__":
    # Define paths
    base_dir = "C:/Users/91826/PycharmProjects/fakeaudiodetection"
    features_dir = os.path.join(base_dir, "processed_features")
    save_dir = os.path.join(base_dir, "saved_model")
    unseen_features_path = os.path.join(features_dir, "1111111unseen_features.npy")
    unseen_labels_path = os.path.join(features_dir, "1111111unseen_labels.npy")
    unseen_mapping_path = os.path.join(features_dir, "1111111unseen_file_mapping.npy")
    model_path = os.path.join(save_dir, "mixed_audio_cnn_gru.pth")

    # Verify model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run cnn_gru_train.py first.")
        exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    unseen_dataset = FeatureDataset(unseen_features_path, unseen_labels_path, unseen_mapping_path)
    unseen_loader = DataLoader(unseen_dataset, batch_size=32)

    # Initialize model and load weights
    model = CNNGRUModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate the model with file-level majority voting and ROC curves
    evaluate_model(model, unseen_loader, unseen_dataset.mapping, device)