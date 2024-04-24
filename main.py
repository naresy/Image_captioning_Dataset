import os
import json
from PIL import Image
import torchvision.transforms as transforms

# Define paths
image_dir = "images"
caption_file = "caption/file.json"

# Define transformations for images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])


# Function to load captions from a JSON file
def load_captions(file_path):
    with open(file_path, "r") as f:
        captions = json.load(f)
    return captions

# Function to preprocess images
def preprocess_images(image_dir, image_transform):
    image_data = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            print("Processing image:", image_path)  # Print image path
            image = Image.open(image_path).convert("RGB")
            image = image_transform(image)
            image_data.append((filename, image))
    return image_data

# Function to create caption dataset
def create_caption_dataset(image_dir, caption_file, image_transform):
    captions = load_captions(caption_file)
    image_data = preprocess_images(image_dir, image_transform)
    
    caption_dataset = []
    for image_filename, image_tensor in image_data:
        if image_filename in captions:
            image_captions = captions[image_filename]
            for caption in image_captions:
                caption_dataset.append((image_tensor, caption))
    
    return caption_dataset

# Example usage
caption_dataset = create_caption_dataset(image_dir, caption_file, image_transform)


def save_to_txt(caption_dataset, txt_file):
    with open(txt_file, "w") as f:
        for _, captions in caption_dataset:
            for caption in captions:
                f.write(caption )  # Write each caption as a whole sentence on a separate line
            f.write("\n")  # Add an extra newline between image captions

# Example usage
txt_file = "new_dataset/result.txt"
save_to_txt(caption_dataset, txt_file)
print("Caption dataset saved to:", txt_file)