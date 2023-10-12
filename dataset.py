import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class ProductDataset(Dataset):
    def __init__(self, target_transform=None):
        super().__init__()
        csv_file = "training_data.csv"
        img_dir = "C:/Users/USER/OneDrive - Teesside University/Documents/Ai Core/MyProject/Facebook Marketplace's Recommendation Ranking System/newcleaned_images/"
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                            transforms.ToTensor()
])
        self.target_transform = target_transform

        categories = ['Home & Garden', 'Baby & Kids Stuff', 'DIY Tools & Materials',
        'Music, Films, Books & Games', 'Phones, Mobile Phones & Telecoms',
        'Clothes, Footwear & Accessories', 'Other Goods',
        'Health & Beauty', 'Sports, Leisure & Travel', 'Appliances',
        'Computers & Software', 'Office Furniture & Equipment',
        'Video Games & Consoles']

        self.encoder = {category: label for label, category in enumerate(categories)}
        self.decoder = {label: category for category, label in self.encoder.items()}

    def __getitem__(self, index):
        img_path = (self.img_dir + self.img_labels.iloc[index, 0] + ".jpg")       
        image = Image.open(img_path)
        label = self.img_labels.iloc[index, 1]
        #label = self.decoder[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def __len__(self):
        #print((self.img_labels).dtypes)
        return len(self.img_labels)


if __name__ == '__main__':
    dataset = ProductDataset()
    print(len(dataset))
    print(dataset[786])
    


