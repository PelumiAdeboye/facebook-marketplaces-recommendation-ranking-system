import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureExtractionCNN(nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractionCNN, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove the last classification layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Add a new fully connected layer with 1000 neurons
        self.feature_fc = nn.Linear(2048, 1000) 

    def forward(self, imgs):
        features = self.resnet50(imgs)
        features = features.view(features.size(0), -1)
        features = self.feature_fc(features)
        return features

def main():
    feature_model = FeatureExtractionCNN()
    torch.save(feature_model.state_dict(), "final_model/image_modelDav.pt")

if __name__ == "__main__":
    main()
