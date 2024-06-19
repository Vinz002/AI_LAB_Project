import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from model import CarBrandModel
from PIL import Image
import string
import easyocr

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the CSV file for label mapping
df = pd.read_csv("data_labels.csv")
label_to_idx = {label: idx for idx, label in enumerate(df["brand"].unique())}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}


# Define the same transformations used during training
transform = transforms.Compose(
    [
        transforms.Resize((800, 600)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

num_classes = len(label_to_idx)
# Load a pre-trained MobileNetV2 model
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
# Move model to the device
model = mobilenet.to(device)
model.load_state_dict(torch.load("86new.pth"))
# If you use cpu:
# model.load_state_dict(torch.load('best_model.pth', map_location=torch.device("cpu")))
model.eval()

# Initialize the OCR reader
reader = easyocr.Reader(["en", "de", "fr", "es", "it", "da"], gpu=True)

# Allowed list
ALLOWED_LIST = string.ascii_uppercase + string.digits


def predict_car_brand_and_license_plate(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR to grayscale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to a PIL Image
    pil_image = Image.fromarray(image_rgb)
    # Apply the defined transformations
    transformed_image = transform(pil_image)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
    transformed_image = transformed_image.to(device)

    # Predict car brand
    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = idx_to_label[predicted.item()]

    # Detect license plate region using EasyOCR
    results = reader.readtext(
        image, allowlist=ALLOWED_LIST, detail=0, paragraph=False, contrast_ths=0.1
    )
    license_plate_text = None
    for result in results:
        if len(result) > 5:  # Assuming license plate has at least 5 characters
            license_plate_text = result
            break

    return predicted_label, license_plate_text, image_rgb


if __name__ == "__main__":
    # Example usage
    image_path = "test5.jpg"
    predicted_brand, license_plate_text, result_image = (
        predict_car_brand_and_license_plate(image_path)
    )
    print(f"The predicted car brand is: {predicted_brand}")
    if license_plate_text:
        print(f"The recognized license plate is: {license_plate_text}")
    else:
        print("License plate not found")

    plt.imshow(result_image)
    plt.axis("off")  # Hide axis
    plt.show()
