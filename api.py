import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
import torchvision.models as models
import faiss

import numpy as np


# Import your image processing script here 
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(), # This also changes the pixels to be in range [0, 1] from [0, 255].
                transforms.Resize((64,64)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    )

    @staticmethod
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        image = self.transform(image)
        image = torch.unsqueeze(image, 0)   # Add the leading '1' dimension at the start of the tensor.
        return image             


class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove the last classification layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Add a new fully connected layer with 1000 neurons
        self.feature_fc = nn.Linear(2048, 1000) 

        self.decoder = decoder

    def forward(self, image):
        features = self.resnet50(image)
        features = features.view(features.size(0), -1)
        features = self.feature_fc(features)
        return features

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

try:
    feature_model = FeatureExtractor()
    model_weights_path = "image_modelDav.pt"
    feature_model.load_state_dict(torch.load(model_weights_path))

    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    with open('file_ids2.pkl', 'rb') as handle2:
        ids = pickle.load(handle2)

    # Load the Faiss index using Faiss's built-in function
    index = faiss.read_index('index2.pkl')


    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    image_processor = ImageProcessor()
    processed_image = image_processor(pil_image)
    image_embedding = feature_model.predict(processed_image)
    
    image_embedding_numpy = image_embedding.detach().numpy()

    return JSONResponse(content={
    "Features": image_embedding_numpy.tolist(),  # Convert to a JSON-serializable format
    })

    
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    
    pil_image = Image.open(image.file)

    image_processor = ImageProcessor()
    processed_image = image_processor(pil_image)
    image_embeddings = feature_model.predict(processed_image)


    n_neighbors = 10
    D, I = index.search(np.array(image_embeddings, dtype=np.float32), n_neighbors)
    similar_index = [ids[i] for i in I[0]] #ids not embeddings

    return JSONResponse(content={
    "similar_index": similar_index, # Return the index of similar images here
        })
    

if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)