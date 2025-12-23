import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model_definitions import ModelA_CNN, ModelB_MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2) Datasets & DataLoaders
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 3) Training function
def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    model.to(device)

    for epoch in range(epochs):
        # Train
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validate
        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss = val_loss_sum / len(test_loader)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return train_acc_history, val_acc_history, train_loss_history, val_loss_history


# 4) Train both models
modelA = ModelA_CNN()
modelB = ModelB_MLP()

print("Training Model A (CNN)")
train_acc_A, val_acc_A, train_loss_A, val_loss_A = train_model(modelA, train_loader, test_loader, epochs=5)

print("Training Model B (MLP)")
train_acc_B, val_acc_B, train_loss_B, val_loss_B = train_model(modelB, train_loader, test_loader, epochs=5)

# 5) Pick best model by final validation accuracy
best_model = modelA if val_acc_A[-1] > val_acc_B[-1] else modelB
best_name = "ModelA_CNN" if best_model is modelA else "ModelB_MLP"
print(f"Best model: {best_name}")

# 6) Save model and name
torch.save(best_model.state_dict(), "mnist_model.pth")
with open("best_model_name.txt", "w") as f:
    f.write(best_name)
print("Saved mnist_model.pth and best_model_name.txt")

# 7) Plot and save curves
plt.figure()
plt.plot(train_acc_A, label="Train Acc (CNN)")
plt.plot(val_acc_A, label="Val Acc (CNN)")
plt.plot(train_acc_B, label="Train Acc (MLP)")
plt.plot(val_acc_B, label="Val Acc (MLP)")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curves.png")

plt.figure()
plt.plot(train_loss_A, label="Train Loss (CNN)")
plt.plot(val_loss_A, label="Val Loss (CNN)")
plt.plot(train_loss_B, label="Train Loss (MLP)")
plt.plot(val_loss_B, label="Val Loss (MLP)")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png")

print("Saved accuracy_curves.png and loss_curves.png")
