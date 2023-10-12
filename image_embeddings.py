import torch
import json
import os

from feature_extractionCNN import FeatureExtractionCNN
from dataset import ProductDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Assuming you have the FeatureExtractionCNN model defined

# Load the pre-trained model
feature_model = FeatureExtractionCNN()
model_weights_path = "final_model/image_modelDav.pt"
feature_model.load_state_dict(torch.load(model_weights_path))
feature_model.eval()

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
train_dataset = ProductDataset()
image_embeddings = {}

# Iterate through the dataset and extract embeddings
for idx in tqdm(range(len(train_dataset))):
    image_id = train_dataset.img_labels.iloc[idx, 0]
    image_path = os.path.join(train_dataset.img_dir, image_id + ".jpg")
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        embedding = feature_model(image)
    
    image_embeddings[image_id] = embedding.tolist()

# Save the embeddings dictionary as JSON
with open("image_embeddings.json", "w") as f:
    json.dump(image_embeddings, f)
