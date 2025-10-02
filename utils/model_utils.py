import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
# from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch_optimizer as optim
from sklearn.model_selection import StratifiedKFold


class MelanomaClassifier(nn.Module):
    """
    Simple MLP model for clinical images only.
    Uses ResNet18 backbone to generate embeddings
    and passes them to an MLP for classification.
    """
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # CNN backbone (ResNet18)
        convnext = models.convnext_tiny(pretrained=True)
        convnext.classifier = nn.Identity()
        self.backbone = convnext
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # resnet.fc = nn.Identity()  # Remove classifier head
        # self.backbone = resnet
        
        # MLP on top of clinical embeddings
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)  # Binary classification
        )

    def forward(self, clinical):
        x = self.backbone(clinical)  # (B, 512)
        x = self.global_pool(x)           # [B, 768, 1, 1]
        x = x.view(x.size(0), -1)         # [B, 768]
        return self.classifier(x).squeeze(1)  # (B,)


# --- Training Function ---
def train_one_epoch(model, loader, optimizer, criterion, device):  # default
    model.train()
    for batch in loader:
        clinical = batch["clinical"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(clinical)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")


# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        clinical = batch["clinical"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(clinical)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits)
    preds = (probs > 0.5).int()

    accuracy = accuracy_score(all_labels, preds)
    try:
        roc_auc = roc_auc_score(all_labels, probs)
    except:
        roc_auc = float("nan")  # Handle single-class case

    try:
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
    except:
        precision = float("nan")
        recall = float("nan")

    loss = F.binary_cross_entropy(probs, all_labels)
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }


@torch.no_grad()
def evaluate_with_preds(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    all_images = []

    for batch in loader:
        clinical = batch["clinical"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(clinical)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_images.extend(clinical.cpu())  # Save images for visualization

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits).squeeze()
    preds = (probs > 0.5).int()

    accuracy = accuracy_score(all_labels, preds)
    try:
        roc_auc = roc_auc_score(all_labels, probs)
    except:
        roc_auc = float("nan")

    try:
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
    except:
        precision = float("nan")
        recall = float("nan")

    loss = F.binary_cross_entropy(probs, all_labels)

    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "preds": preds.tolist(),
        "labels": all_labels.int().tolist(),
        "images": all_images,  # List of image tensors
        "probs": probs.tolist()
    }