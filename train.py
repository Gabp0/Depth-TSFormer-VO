import os
import glob
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
from json import dump
from tqdm import tqdm
from datetime import datetime

from model import SiamVO
from kitti_dataset import KITTIOdometryDataset, RelativePoseLoss

def train(batch_size, epochs, transform, learning_rate, dataset_root):

    train_data = KITTIOdometryDataset(dataset_root, '00.txt', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    validation_data = KITTIOdometryDataset(dataset_root, '01.txt', transform=transform)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamVO().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = RelativePoseLoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for img1, img2, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for img1, img2, target in validation_loader:
                img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                output = model(img1, img2)
                loss = criterion(output, target)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(validation_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            model.train()

        # Save model checkpoint
        checkpoint_path = f"siamvo_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    batch_size = 16
    epochs = 20
    learning_rate = 0.001
    dataset_root = "dataset"

    transform = transforms.Compose([
        transforms.Resize((128, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train(batch_size, epochs, transform, learning_rate, dataset_root)