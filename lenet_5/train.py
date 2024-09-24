import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

from model import LeNet

# Set seed for reproducibility
torch.manual_seed(42)

# Enhanced Data augmentation and normalization
transform_train = transforms.Compose([
    # Rotate images randomly by up to 20 degrees
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(
        0.15, 0.15)),  # Randomly translate images up to 15%
    # Random perspective transform
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),  # Adjust color properties
    transforms.RandomCrop(28, padding=4),  # Crop images randomly
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    # Randomly flip images vertically (might be less useful for digits)
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the LeNet-5 model and optimizer
model = LeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training is done on : ", device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/LeNet-MNIST')

# Create a directory for saving models
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Function to save the model


def save_model(model, epoch, path="saved_models/lenet_mnist.pth"):
    print(f"Saving model at epoch {epoch}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    print(f"> Epoch {epoch} <\nLoss: {
          train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Log training loss and accuracy to TensorBoard
    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)

    return train_loss, train_accuracy


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {
          test_accuracy:.4f}\n-----------")

    # Log test loss and accuracy to TensorBoard
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)

    return test_loss, test_accuracy


# Training loop
num_epochs = 15
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
best_accuracy = 0.0  # For saving the best model

for epoch in range(num_epochs):
    train_loss, train_acc = train(
        model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion, epoch)

    # Store losses and accuracies
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # Step the scheduler
    scheduler.step()

    # Save model if the test accuracy is improved
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        save_model(model, epoch)

# Close the TensorBoard writer after training
writer.close()

# Plot training & testing loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, num_epochs+1), test_accuracies, label="Test Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()
