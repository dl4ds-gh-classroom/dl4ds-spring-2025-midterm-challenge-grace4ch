import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
import eval_cifar100
import eval_ood
import os

# ==========================
# Model Definition: ResNet50
# ==========================
class PretrainedModel(nn.Module):
    def __init__(self, num_classes=100):
        super(PretrainedModel, self).__init__()
        # Load pretrained ResNet50 model
        self.base_model = timm.create_model("resnet50", pretrained=True)
        in_features = self.base_model.get_classifier().in_features
        
        # Replace the classifier with dropout and linear layer
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# ==========================
# Mixup for Regularization
# ==========================
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================
# Training Loop
# ==========================
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    model.train()
    device = CONFIG["device"]
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if CONFIG.get("mixup_alpha", 0) > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG["mixup_alpha"], device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix({"loss": running_loss / (total / CONFIG["batch_size"]), "acc": 100. * correct / total})

    return running_loss / len(trainloader), 100. * correct / total

# ==========================
# Validation Loop
# ==========================
def validate(model, valloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": running_loss / (total / valloader.batch_size), "acc": 100. * correct / total})
    return running_loss / len(valloader), 100. * correct / total

# ==========================
# Main Function
# ==========================
def main():
    CONFIG = {
        "model": "ResNet50-Transfer-Attempt7",
        "batch_size": 128,
        "learning_rate": 5e-4,
        "epochs": 80,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "seed": 42,
        "mixup_alpha": 0.2,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1
    }

    # Reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed(CONFIG["seed"])

    # Data augmentations for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    # Standard normalization for testing/validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Dataset split
    trainset = torchvision.datasets.CIFAR100(CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    valset.dataset.transform = transform_test

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    testset = torchvision.datasets.CIFAR100(CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # Load model
    model = PretrainedModel(num_classes=100).to(CONFIG["device"])

    # Fully unfreeze from start
    for param in model.base_model.parameters():
        param.requires_grad = True

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["learning_rate"] / 100)

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    # ========================
    # Final Test + CSV Export
    # ========================
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    df = eval_ood.create_ood_df(all_predictions)
    df.to_csv("submission_part3_attempt7.csv", index=False)
    print("submission_part3_attempt7.csv created successfully.")

if __name__ == '__main__':
    main()
