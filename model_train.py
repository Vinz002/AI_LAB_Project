import pandas as pd
from torchvision import transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import CarBrandDataset
import torch
import torch.nn as nn
import torch.optim as optim

# Load the CSV file
df = pd.read_csv("data_labels.csv")

# Separate into train and test dataframes
train_df = df[df["set"] == "train"]
test_df = df[df["set"] == "test"]

# Debugging: Check the size of the datasets
print(f"Number of training samples: {len(train_df)}")
print(f"Number of test samples: {len(test_df)}")

# Compute class weights
class_counts = train_df["brand"].value_counts().to_dict()
class_weights = {label: 1.0 / count for label, count in class_counts.items()}
weights = train_df["brand"].map(class_weights).values

sampler = WeightedRandomSampler(
    weights=weights, num_samples=len(weights), replacement=True
)

# print(f"Class counts: {class_counts}")
# print(f"Class weights: {class_weights}")

# Define transforms
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.75),
            transforms.Resize((800, 600)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((800, 600)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Create datasets
train_dataset = CarBrandDataset(train_df, transform=data_transforms["train"])
test_dataset = CarBrandDataset(test_df, transform=data_transforms["test"])

# Debugging: Check the first few items in the datasets
# print(f"First training sample: {train_dataset[0]}")
# print(f"First test sample: {test_dataset[0]}")

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=48, sampler=sampler, pin_memory=True
)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, pin_memory=True)

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load a pre-trained MobileNetV2 model
mobilenet = models.mobilenet_v2(pretrained=True)

# Modify the classifier to match the number of classes
num_classes = len(train_df["brand"].unique())
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)

# Freeze the initial layers except the last few
for name, param in mobilenet.named_parameters():
    if (
        "features.18" in name
        or "features.17" in name
        or "features.16" in name
        or "features.15" in name
        or "classifier" in name
    ):  # The last blocks and classifier
        param.requires_grad = True
        print(f"Unfreezing: {name}")
    else:
        param.requires_grad = False
        print(f"Freezing: {name}")

# Move model to the device
model = mobilenet.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [
        {"params": mobilenet.classifier[1].parameters(), "lr": 0.0005},
        {"params": mobilenet.features[18].parameters(), "lr": 0.0001},
        {"params": mobilenet.features[17].parameters(), "lr": 0.0001},
        {"params": mobilenet.features[16].parameters(), "lr": 0.0001},
        {"params": mobilenet.features[15].parameters(), "lr": 0.0001},
    ]
)


# Train the model
num_epochs = 15
best_val = 0
index = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = None
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        index += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = 0.7 * running_loss + 0.3 * loss.item()

        # Debugging: Print progress for every 100 batches
        if index % 500 == 0:
            print(" ")
            # Compute val loss and print it
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            if val_loss < best_val or best_val == 0:
                best_val = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(
                    f"Best model saved with loss: {val_loss / len(test_loader.dataset):.4f}"
                )
            accuracy = correct / total
            print(
                f"Val loss: {val_loss / len(test_loader.dataset):.4f} ({accuracy * 100:.2f}%)"
            )
        if index % 100 == 0:
            print(" ")
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss:.4f}"
            )
        if index % 10 == 0:
            print(".", end="", flush=True)
        # torch.cuda.empty_cache()

# load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test accuracy: {accuracy * 100:.2f}%")
