import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageTransferCNN(nn.Module):
    def __init__(self, num_classes=13) -> None:
        super(ImageTransferCNN, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, imgs):
        return F.softmax(self.resnet50(imgs), dim=1)

def main():
    model = ImageTransferCNN()
    print(model)

if __name__ == '__main__':
    main()
