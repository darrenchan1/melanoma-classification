import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.transforms.functional as TF


def plot_confusion_matrix(preds, labels, class_names=("Benign", "Malignant")):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


def plot_auc_curve(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean


def show_predictions(results, max_images=25, class_names=("Benign", "Malignant")):
    preds = results["preds"]
    labels = results["labels"]
    images = results["images"]
    probs = results["probs"]

    n = min(len(images), max_images)
    cols = 5
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    for i in range(n):
        img = images[i]
        pred = preds[i]
        label = labels[i]
        prob = probs[i]

        img = denormalize(img).clamp(0, 1)
        img = TF.to_pil_image(img)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

        correct = pred == label
        color = "green" if correct else "red"
        title = f"Pred: {class_names[pred]} ({prob:.2f})\nTrue: {class_names[label]}"
        plt.title(title, color=color, fontsize=10)

    plt.tight_layout()
    plt.show()
