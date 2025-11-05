import numpy as np
import random
import torch
import torchvision 
import torchsampler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchsampler.imbalanced import ImbalancedDatasetSampler
from sklearn.metrics import classification_report


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dir = './train_images'
test_dir = './test_images'
print('Finished loading libs')

transform = transforms.Compose([
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

train_sampler = ImbalancedDatasetSampler(
    train_data,
    indices=train_idx,
    callback_get_label=get_label
)

valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,   # deterministic order
    num_workers=0
)

print('Finished loading data')


classes = ('noface', 'face')


class Net2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()   # <-- fixed (was: super(Net, self).__init__())
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # /2

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # /4

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # global spatial pooling
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



net = Net2().to(device)


print('Training net')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

num_epochs = 4

for epoch in range(num_epochs):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

net.eval()

all_preds = []
all_labels = []

# --- Collect predictions and labels ---
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Convert to numpy arrays ---
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Compute accuracy ---
accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
print(f'Accuracy of the network on test images: {accuracy:.2f} %')

# --- Classification Report ---
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes, digits=3))


import matplotlib.pyplot as plt



num_classes = len(classes)
cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

cm_np = cm.cpu().numpy()
print("Confusion matrix (rows=true, cols=predicted):\n", cm_np)

# plot matrix with counts and row-wise percentages
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm_np, cmap="Blues")
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion matrix")

row_sums = cm_np.sum(axis=1, keepdims=True)
thresh = cm_np.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        pct = (cm_np[i, j] / row_sums[i, 0]) if row_sums[i, 0] > 0 else 0.0
        text = f"{cm_np[i,j]}\n{pct:.1%}"
        ax.text(j, i, text, ha="center", va="center",
                color="white" if cm_np[i, j] > thresh else "black")

plt.tight_layout()
plt.show()