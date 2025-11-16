## This script is for training the binary classifer.

import numpy as np
import torch
import torchvision 
import torchsampler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from deep_learning_project.torchsampler.imbalanced import ImbalancedDatasetSampler
from face_noface_networks import Net2, Net3
import time 
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (note: may slow things down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
train_dir = '/Users/alberthogsted/Desktop/DTU/5. Semester/Machine Learning and Data Analytics/Scripts/deep_learning_project/train_images'
test_dir = '/Users/alberthogsted/Desktop/DTU/5. Semester/Machine Learning and Data Analytics/Scripts/deep_learning_project/test_images'
print('Finished loading libs')

transform = transforms.Compose([
    transforms.Resize((32, 32)),   # ensure consistent input size for the network
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,), std=(1,))
])

# --- Load Data ---
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

# --- Split Train into Train + Validation ---
valid_size = 0.2
batch_size = 32

num_train = len(train_data)
indices = np.arange(num_train)
np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))
valid_idx, train_idx = indices[:split], indices[split:]

# --- Define Samplers using ImbalancedDatasetSampler ---
def get_label(dataset, idx):
    return dataset[idx][1]

train_sampler = ImbalancedDatasetSampler(train_data, indices=train_idx, callback_get_label=get_label)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

# --- Data Loaders ---
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)



# --- Classes ---
classes = ('noface', 'face')


net = Net3()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training hyperparams
max_epochs = 50         # upper bound; adjust based on dataset
patience = 7            # stop if no improvement for this many epochs
best_val_loss = np.inf
epochs_no_improve = 0
best_epoch = 0
PATH = './net3_flipped.pth'

print("Training net (with early stopping)...")
start_time = time.time()
for epoch in range(1, max_epochs + 1):
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.sampler) if hasattr(train_loader, "sampler") else running_loss / len(train_loader.dataset)

    # validation
    net.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(valid_loader.sampler) if hasattr(valid_loader, "sampler") else len(valid_loader.dataset)
    val_acc = val_correct / (val_total + 1e-12)

    print(f"Epoch {epoch:03d}  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")

    # checkpoint / early stopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_epoch = epoch
        torch.save(net.state_dict(), PATH)
        print(f"  Saved best model (epoch {epoch})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after epoch {epoch}. Best epoch: {best_epoch}")
            break

elapsed = time.time() - start_time
print(f"Training finished in {elapsed:.1f}s; best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
# ...existing code...

correct = 0
total = 0


with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test images: {100 * correct // total} %')
