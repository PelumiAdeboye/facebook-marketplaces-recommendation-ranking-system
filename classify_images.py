from dataset import ProductDataset
from ResNet50_CNN import ImageTransferCNN

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
#from sklearn.model_selection import train_test_split

# Define your dataset
dataset = ProductDataset()

# Split the dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define dataloaders for each set
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)



def train(model, epochs, train_dataloader, val_dataloader, fine_tune=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter()
    batch_idx = 0

    if fine_tune:
        # Get all the layers of the model
        all_layers = list(model.resnet50.children())
            
        # Set requires_grad to True for the last two layers
        for layer in all_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Optimize only unfrozen layers
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # Define folder structure for model evaluation
    model_eval_folder = "model_evaluation"
    if not os.path.exists(model_eval_folder):
        os.mkdir(model_eval_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    model_folder = os.path.join(model_eval_folder, f"model_{timestamp}")
    os.mkdir(model_folder)

    weights_folder = os.path.join(model_folder, "weights")
    os.mkdir(weights_folder)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}, Training Loss: {train_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                writer.add_scalar("Validation Loss", val_loss, batch_idx)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Save model weights at the end of each epoch
        weight_filename = f"weights_epoch_{epoch+1}.pt"
        weight_filepath = os.path.join(weights_folder, weight_filename)
        torch.save(model.state_dict(), weight_filepath)
                
        

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    model = ImageTransferCNN()
    train(model, epochs=1, train_dataloader=train_loader, val_dataloader=val_loader, fine_tune=True)


