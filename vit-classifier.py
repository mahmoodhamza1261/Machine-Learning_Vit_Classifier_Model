import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set directory path
data_dir = "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset"  # The path to your dataset folder

# Define the image transforms (for data augmentation and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT requires a 224x224 input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for ViT
])

# Load dataset using ImageFolder (for folder structure with labels as subfolder names)
# dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into training and validation sets (optional, but recommended)
# from torch.utils.data import random_split
# train_size = int(0.6 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_dataset="/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_train/laplace_classification_dataset_train"
# val_dataset="/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_val/laplace_classification_dataset_val"

train_dataset = datasets.ImageFolder(
    root="/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_train/laplace_classification_dataset_train",
    transform=transform
)
val_dataset = datasets.ImageFolder(
    root="/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_val/laplace_classification_dataset_val",
    transform=transform
)


# DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Classes in the dataset: {train_dataset.classes}")  # It should output ['not_product', 'product']




import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Load the ViT model from Hugging Face (pre-trained on ImageNet)
model_name = "google/vit-large-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name)

# Replace the final classification head to match your number of classes
num_classes = 11  # In your case, the number of classes (e.g., "product" vs. "not product")
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# Define device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)





import torch.optim as optim
import torch.nn as nn

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for batch in train_dataloader:
        # Get inputs and labels
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Average loss per epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_dataloader)}")







model.eval()  # Set model to evaluation mode
correct = 0
total = 0

# No gradients required for evaluation
with torch.no_grad():
    for batch in val_dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs).logits
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Validation Accuracy: {accuracy:.2f}%")



# Save entire model (including architecture)
# Save the trained model
model_save_path = "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/save_model/trained_vit_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# # Load the entire model
# # Recreate the model (same architecture as before)
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
model.classifier = torch.nn.Linear(model.classifier.in_features, 11)  # Adjust the number of classes

# Load the saved model weights
model_load_path = "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/save_model/trained_vit_model.pth"
model.load_state_dict(torch.load(model_load_path))
model.to(device)  # Move the model to the correct device (CPU or GPU)

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully")




# from PIL import Image
# from transformers import ViTImageProcessor

# # Load processor (pre-trained on ImageNet)
# processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

# def classify_image(image_path, model, processor, device):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(device)

#     # Forward pass to get logits and predictions
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)

#     # Get the predicted class (the one with highest probability)
#     predicted_class = probs.argmax().item()
#     confidence = probs[0][predicted_class].item()

#     return predicted_class, confidence

# # Test on an image
# image_path = "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/SA-01-Y/2_11.jpg"  # Path to a new image you want to classify
# predicted_class, confidence = classify_image(image_path, model, processor, device)

# print(f"Predicted class: {predicted_class}, Confidence: {confidence}")



#test on mutiple images in test folder
import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class_mapping = {
    "LEI-01": 0, "LEM-01-B": 1, "LEM-01-O": 2, "LEM-01-W": 3, "LEM-01-Y": 4,
    "PB-01-G": 5, "PB-01-L": 6, "PB-01-P": 7, "SA-01-P": 8, "SA-01-R": 9, "SA-01-Y": 10
}

# Load model and adjust for the number of classes
model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
model.classifier = torch.nn.Linear(model.classifier.in_features, 11)  # Adjust number of classes

# Load the saved model weights
model_load_path = "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/save_model/trained_vit_model.pth"
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to(device)  # Move the model to the correct device (CPU or GPU)

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully")

# Load processor
processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

# Define a function for image classification
def classify_image(image_path, model, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = probs.argmax().item()
        confidence = probs[0][predicted_class].item()
        return predicted_class, confidence
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

# List of directories containing images
image_directories = [
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/LEI-01",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/LEM-01-B",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/LEM-01-O",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/LEM-01-W",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/LEM-01-Y",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/PB-01-G",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/PB-01-L",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/PB-01-P",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/SA-01-P",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/SA-01-R",
    "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/laplace_classification_dataset/laplace_classification_dataset_test/classification_dataset_train/SA-01-Y",
    # Add more directories as needed
]

# Prepare lists for storing true labels and predictions
true_labels = []
predictions = []

# Iterate through each directory and classify images
all_results = []
for directory in image_directories:
    print(f"\nProcessing directory: {directory}")
    if os.path.isdir(directory):  # Ensure it's a valid directory
        directory_name = directory.split('/')[-1]  # Extract directory name (e.g., "LEI-01")
        true_label = class_mapping.get(directory_name, None)  # Get true label from mapping
        
        if true_label is not None:  # Only process if the directory name matches a class in the mapping
            for image_name in os.listdir(directory):
                image_path = os.path.join(directory, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
                    predicted_class, confidence = classify_image(image_path, model, processor, device)
                    if predicted_class is not None:  # Ensure classification was successful
                        predictions.append(predicted_class)
                        true_labels.append(true_label)
                        all_results.append((directory_name, image_name, predicted_class, confidence))
                        print(f"Image: {image_name}, Predicted class: {predicted_class}, Confidence: {confidence}")

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_mapping.keys()))

# Display the confusion matrix
cm_display.plot(cmap='Blues')

# Compute TP, TN, FP, FN for each class
def calculate_metrics(cm):
    metrics = {}
    num_classes = cm.shape[0]
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
    return metrics

metrics = calculate_metrics(cm)

# Print TP, TN, FP, FN for each class
for idx, class_name in enumerate(class_mapping.keys()):
    print(f"\nClass: {class_name}")
    print(f"True Positives (TP): {metrics[idx]['TP']}")
    print(f"True Negatives (TN): {metrics[idx]['TN']}")
    print(f"False Positives (FP): {metrics[idx]['FP']}")
    print(f"False Negatives (FN): {metrics[idx]['FN']}")











#another code for calculation evaluation matrix
def classify_image(image_path, model, processor, device, k=1):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Forward pass to get logits and predictions
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the top K predictions
    _, top_k_predictions = torch.topk(logits, k, dim=1)
    top_k_predictions = top_k_predictions.squeeze().cpu().numpy()

    return top_k_predictions

def top_k_accuracy(top_k_predictions, true_label, k=1):
    # Check if true label is in top K predictions
    return true_label in top_k_predictions

# Set the top-k value (for example, 3 for Top-3 accuracy)
k = 3

# Initialize lists to store true labels and predictions
true_labels = []
predictions = []
top_k_accuracies = []  # Store Top-K accuracy values for each image

# Iterate through each directory and classify images
all_results = []
for directory in image_directories:
    print(f"\nProcessing directory: {directory}")
    if os.path.isdir(directory):  # Ensure it's a valid directory
        directory_name = directory.split('/')[-1]  # Extract directory name (e.g., "LEI-01")
        true_label = class_mapping.get(directory_name, None)  # Get true label from mapping
        
        if true_label is not None:  # Only process if the directory name matches a class in the mapping
            for image_name in os.listdir(directory):
                image_path = os.path.join(directory, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
                    top_k_predictions = classify_image(image_path, model, processor, device, k)
                    if top_k_predictions is not None:  # Ensure classification was successful
                        predictions.append(top_k_predictions[0])  # We store the top-1 prediction for confusion matrix
                        true_labels.append(true_label)
                        
                        # Check if the true label is within the top K predictions
                        is_correct = top_k_accuracy(top_k_predictions, true_label, k)
                        top_k_accuracies.append(is_correct)
                        
                        all_results.append((directory_name, image_name, top_k_predictions, is_correct))
                        print(f"Image: {image_name}, Predicted top-{k} classes: {top_k_predictions}, True label: {true_label}, Correct: {is_correct}")

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_mapping.keys()))

# Display the confusion matrix
cm_display.plot(cmap='Blues')

# Compute TP, TN, FP, FN for each class
def calculate_metrics(cm):
    metrics = {}
    num_classes = cm.shape[0]
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
    return metrics

metrics = calculate_metrics(cm)

# Print TP, TN, FP, FN for each class
for idx, class_name in enumerate(class_mapping.keys()):
    print(f"\nClass: {class_name}")
    print(f"True Positives (TP): {metrics[idx]['TP']}")
    print(f"True Negatives (TN): {metrics[idx]['TN']}")
    print(f"False Positives (FP): {metrics[idx]['FP']}")
    print(f"False Negatives (FN): {metrics[idx]['FN']}")

# Calculate Top-K accuracy
top_k_accuracy_percentage = (sum(top_k_accuracies) / len(top_k_accuracies)) * 100
print(f"\nTop-{k} Accuracy: {top_k_accuracy_percentage:.2f}%")



#__________________________extracting boundary boxes from image for each product_______________________________________________


# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor

# image = cv2.imread("/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/base_imgs/img_303.jpg")


# resized_image = cv2.resize(image, (756, 1008))  # Resize to manageable dimensions
# sam = sam_model_registry["vit_b"](checkpoint="/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/sam_vit_b_01ec64.pth")
# sam.to(device="cuda")
# print("Model Loaded")
# mask_generator = SamAutomaticMaskGenerator(sam,pred_iou_thresh=0.86)
# masks = mask_generator.generate(resized_image)

# # show_anns(masks)
# output_image = resized_image.copy()
# for mask in masks:
#     # Extract the bounding box (XYWH format)
#     bbox = mask['bbox']
#     x, y, w, h = map(int, bbox)  # Convert to integers

#     # Draw the bounding box on the image (using a random color)
#     color = (0, 255, 0)  # Green color for bounding box
#     thickness = 2  # Thickness of the bounding box line
#     cv2.rectangle(output_image, (x, y), (x + w, y + h), color, thickness)

# # Save or display the result
# output_path = "/home/ubuntu/Desktop/pipeline_experimentation/infer_sam/output_with_bboxes.jpg"
# cv2.imwrite(output_path, output_image)
















# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image

# # Load the classifier model
# # classifier_model_name = "google/vit-base-patch16-224-in21k"
# # classifier = ViTForImageClassification.from_pretrained(model_name)
# # classifier_processor = ViTImageProcessor.from_pretrained(model_name)
# model.to("cuda")

# # Function to classify a single image
# def classify_bbox_image(image_array, model, processor, device="cuda"):
#     # Convert the NumPy array (BGR from OpenCV) to a PIL image
#     image_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

#     # Preprocess the image
#     inputs = processor(images=image_pil, return_tensors="pt").to(device)
    
#     # Predict using the classifier
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     predicted_class = probs.argmax().item()
#     confidence = probs[0][predicted_class].item()
    
#     return predicted_class, confidence

# # Loop through each mask's bounding box and classify
# for mask in masks:
#     bbox = mask["bbox"]
#     x, y, w, h = map(int, bbox)  # Convert bounding box to integers
    
#     # Crop the bounding box area
#     test_image = resized_image[y:y+h, x:x+w]  # Crop the region of interest (ROI)

#     # Skip invalid crops (e.g., if dimensions are too small)
#     if test_image.size == 0:
#         continue

#     # Classify the cropped bounding box
#     predicted_class, confidence = classify_bbox_image(test_image, model, processor, device="cuda")
#     print(f"Bounding Box: {bbox}, Predicted Class: {predicted_class}, Confidence: {confidence}")




#    # ______________________________________-storing predicted images in sub folders with its classname_________________________
# import os

# # Define output directory for classified images
# output_dir = "/home/ubuntu/Desktop/pipeline_experimentation/classified_images"

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Loop through each mask's bounding box and classify
# for i, mask in enumerate(masks):  # Enumerate for unique naming
#     bbox = mask["bbox"]
#     x, y, w, h = map(int, bbox)  # Convert bounding box to integers
    
#     # Crop the bounding box area
#     test_image = resized_image[y:y+h, x:x+w]  # Crop the region of interest (ROI)

#     # Skip invalid crops (e.g., if dimensions are too small)
#     if test_image.size == 0:
#         continue

#     # Classify the cropped bounding box
#     predicted_class, confidence = classify_bbox_image(test_image, model, processor, device="cuda")
#     print(f"Bounding Box: {bbox}, Predicted Class: {predicted_class}, Confidence: {confidence}")

#     # Save the cropped image into the corresponding class subfolder
#     class_folder = os.path.join(output_dir, str(predicted_class))
#     os.makedirs(class_folder, exist_ok=True)  # Create subfolder for the class if not exists

#     # Save the cropped image
#     image_filename = os.path.join(class_folder, f"bbox_{i}_conf_{confidence:.2f}.jpg")
#     cv2.imwrite(image_filename, test_image)

# print(f"All images saved to {output_dir}")

