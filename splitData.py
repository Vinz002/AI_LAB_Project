import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
data_dir = 'dataset\imgs'
output_csv = 'data_labels.csv'
split_ratio = 0.2  # 20% test, 80% train

# List to hold file information
data = []

# Get all brand folders
brands = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for brand in brands:
    brand_path = os.path.join(data_dir, brand)
    
    # Get all images in the brand folder
    images = [f for f in os.listdir(brand_path) if os.path.isfile(os.path.join(brand_path, f))]
    
    # Split images into train and test
    train_images, test_images = train_test_split(images, test_size=split_ratio, random_state=42)
    
    # Append train images with label 'train'
    for img in train_images:
        data.append([os.path.join(brand_path, img), brand, 'train'])
    
    # Append test images with label 'test'
    for img in test_images:
        data.append([os.path.join(brand_path, img), brand, 'test'])

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=['file_path', 'brand', 'set'])
df.to_csv(output_csv, index=False)

print(f"Data has been split and saved to {output_csv}")
