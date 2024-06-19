import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import CarBrandDataset, CarBrandModel
import torch

# Load the CSV file
df = pd.read_csv('data_labels.csv')

# Separate into train and test dataframes
train_df = df[df['set'] == 'train']
test_df = df[df['set'] == 'test']

# Debugging: Check the size of the datasets
print(f"Number of training samples: {len(train_df)}")
print(f"Number of test samples: {len(test_df)}")

# Compute class weights
class_counts = train_df['brand'].value_counts().to_dict()
class_weights = {label: 1.0/count for label, count in class_counts.items()}
weights = train_df['brand'].map(class_weights).values

sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")

# Define transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((300, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((300, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets
train_dataset = CarBrandDataset(train_df, transform=data_transforms['train'])
test_dataset = CarBrandDataset(test_df, transform=data_transforms['test'])

# Debugging: Check the first few items in the datasets
#print(f"First training sample: {train_dataset[0]}")
#print(f"First test sample: {test_dataset[0]}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


num_classes = len(train_df['brand'].unique())
model = CarBrandModel(num_classes=num_classes).to(device)

# load the best model
model.load_state_dict(torch.load('best_model.pth'))

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
