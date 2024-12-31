# cnn_train.py: Training process

# Import necessary libraries for model training, data manipulation, and visualization
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

from models.cnn import CNN
from get_dataset import GetData

# Set the device to use MPS (MacOS equivalent of CUDA) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Define the transformations for data preprocessing and augmentation
transform = v2.Compose([
    v2.Resize((64, 64)),                    # Resize images to 64x64
    v2.Grayscale(num_output_channels=3),    # Convert images to grayscale with 3 channels
    v2.RandomHorizontalFlip(),              # Randomly flip images horizontally
    
    # Aggressive data augmentations that does not work
    # v2.RandomRotation(degrees=10),  # Random rotations within ±10 degrees
    # v2.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.1)),  # ±10% zoom
    # v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ±10% horizontal/vertical shifting
    
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] # Normalization
    )
])

# Load the training, validation, and test datasets with transformations
train_dataset = GetData(csv_file='data/train_labels.csv',
                         img_dir='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_image, train_label = next(iter(train_loader))

val_dataset = GetData(csv_file='data/valid_labels.csv', 
                       img_dir='data/valid/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
val_image, val_label = next(iter(val_loader))

test_dataset = GetData(csv_file='data/test_labels.csv', 
                        img_dir='data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_image, test_label = next(iter(test_loader))

# Load the CNN model and move it to the selected device
model = CNN().to(device)

# Class Weighting that does not work

# from collections import Counter
# # Get all labels from the dataset
# all_labels = train_dataset.labels.iloc[:, 1].tolist()  # Extract label column
# # Count class frequencies
# class_counts = Counter(all_labels)
# print("Class counts:", class_counts)
# # Calculate class weights
# total_samples = sum(class_counts.values())
# class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
# # Convert to tensor and ensure sorted order by class index
# sorted_weights = [class_weights[cls] for cls in sorted(class_counts.keys())]
# class_weights_tensor = torch.tensor(sorted_weights, dtype=torch.float).to(device)

# Define the loss function (CrossEntropyLoss for classification)
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1, verbose=True)

# Initialize variables for tracking training progress
best_val_acc = 0
epoch_counter = 0

num_epochs = 80

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Start the training loop
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Training phase: iterate through the training data loader
    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = data[0].to(device), data[1].to(device) # Move data to device

        optimizer.zero_grad()               # Reset gradients
        outputs = model(inputs)             # Forward pass
        loss = criterion(outputs, labels)   # Compute loss
        loss.backward()                     # Backward pass
        optimizer.step()                    # Update weights

        # Update training metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Test phase: evaluate on test data
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Store test loss and accuracy
    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # Validation phase: evaluate on validation data
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Store validation loss and accuracy
    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1},
          Train Loss: {train_loss}, Train Accuracy: {train_acc},
          Test Loss: {test_loss}, Test Accuracy: {test_acc},
          Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    
    # Update the scheduler with the validation accuracy
    scheduler.step(val_acc)

    # Log the current learning rate
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']:.6f}")
    
    epoch_counter += 1

    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0 
        torch.save(model.state_dict(), 'model_cnn_fer2013.pth')

# Save training results to a CSV file
df = pd.DataFrame({
    'Epoch': range(1, epoch_counter+1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies,
    'Validation Accuracy': val_accuracies
})
df.to_csv('result_cnn_fer2013.csv', index=False)

# Plot and save accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
plt.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy')
plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.title('Accuracy for CNN on FER2013')
plt.savefig('accuracy_cnn_fer2013.png')

# Plot and save loss graph
plt.figure(figsize=(8, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.title('Loss for CNN on FER2013')
plt.savefig('loss_cnn_fer2013.png')

# Define and save confusion matrix visualization
def plot_confusion_matrix(model, data_loader, classes, device, normalize=False):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix and plot as heatmap
    cm = confusion_matrix(all_labels, all_preds, normalize='true' if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.yticks(rotation=45, ha='right')  # Rotate x-axis labels

    plt.title('Confusion Matrix for CNN on FER2013')
    plt.savefig('cm_cnn_fer2013.png')

# Compute confusion matrix and plot as heatmap
classes = ['happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'neutral']
plot_confusion_matrix(model, test_loader, classes, device, normalize=True)