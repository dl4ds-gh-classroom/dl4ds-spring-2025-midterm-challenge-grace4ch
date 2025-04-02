import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
import wandb
from tqdm.auto import tqdm
import eval_cifar100
import eval_ood

# -----------------------------------------------------------------------------
# Simple CNN
# -----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg(x).view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# -----------------------------------------------------------------------------
# Train for one epoch
# -----------------------------------------------------------------------------
def train(epoch, model, loader, optimizer, criterion, CONFIG):
    model.train()
    device = CONFIG["device"]
    total, correct, loss_sum = 0, 0, 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")

    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        loop.set_postfix(loss=loss_sum/len(loader), acc=100. * correct / total)

    return loss_sum / len(loader), 100. * correct / total

# -----------------------------------------------------------------------------
# Evaluate model on validation or test set
# -----------------------------------------------------------------------------
def validate(model, loader, criterion, CONFIG):
    model.eval()
    device = CONFIG["device"]
    total, correct, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="[Validate]")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss_sum += loss.item()
            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            loop.set_postfix(loss=loss_sum/len(loader), acc=100. * correct / total)

    return loss_sum / len(loader), 100. * correct / total

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    CONFIG = {
        "model": "Part1_SimpleCNN",
        "batch_size": 8,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "epochs": 100,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed(CONFIG["seed"])

    # -------------------------------
    # Transforms and Dataset Loading
    # -------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    dataset = torchvision.datasets.CIFAR100(CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    trainset, valset = random_split(dataset, [train_len, val_len])
    valset.dataset.transform = transform_test

    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True,
                             num_workers=CONFIG["num_workers"], pin_memory=True)
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False,
                           num_workers=CONFIG["num_workers"], pin_memory=True)
    testset = torchvision.datasets.CIFAR100(CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    # -------------------------------
    # Model, Optimizer, and Scheduler
    # -------------------------------
    model = SimpleCNN().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9,
                          weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # -------------------------------
    # wandb Setup
    # -------------------------------
    wandb.login()
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name="part1_simplecnn_attempt1")
    wandb.watch(model)

    # -------------------------------
    # Training Loop
    # -------------------------------
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG)
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    # -------------------------------
    # Final Evaluation + CSV Export
    # -------------------------------
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    df = eval_ood.create_ood_df(all_predictions)
    df.to_csv("submission_part1_attempt1.csv", index=False)
    print("submission_part1_attempt1.csv created successfully.")

if __name__ == "__main__":
    main()
