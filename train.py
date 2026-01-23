import os
import torch
import numpy as np
import random
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from datetime import datetime

# from model_siam import SiamVO
from model import ViTVO
from kitti_dataset import KITTIOdometryDataset
from params import params

import matplotlib.pyplot as plt

def train(params):

    train_sequences = ["00", "02", "08", "09"]
    test_sequences = ['01', '03', '04', '05', '06', '07', '10']

    dataset = KITTIOdometryDataset(2, params["dataset_root"], train_sequences)

    val_len = int(len(dataset) * params["val_split"])
    test_len = int(len(dataset) - val_len)

    # train_data = Subset(dataset, list(range(0, test_len)))
    # val_data = Subset(dataset, list(range(test_len, test_len + val_len)))
    train_data, val_data = random_split(dataset, [test_len, val_len], generator=torch.Generator().manual_seed(99))

    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTVO(params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = torch.nn.MSELoss()

    output_dir = os.path.join(params["output_dir"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    loss_history = []
    lowest_loss = (0, float('inf'))

    for epoch in range(params["epochs"]):
        print(f"Epoch {epoch+1}/{params['epochs']}")
        model.train()
        train_loss = 0.0

        # for imgs, gt in tqdm(train_loader):
        for imgs, gt in train_loader:
            imgs = imgs.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            y = model(imgs)
            loss = criterion(y, gt)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{params['epochs']}], Total Loss: {train_loss:.4f}")

        # validation
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for imgs, gt in val_loader:
                imgs = imgs.to(device)
                gt = gt.to(device)

                y = model(imgs)
                loss = criterion(y, gt)
                
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f"dtsvo_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        loss_history.append((train_loss, val_loss))

        if val_loss < lowest_loss[1]:
            lowest_loss = (epoch+1, val_loss)
            best_model_path = os.path.join(output_dir, f"dtsvo_epoch_{epoch+1}-best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch+1} with validation loss {val_loss:.4f}")

        # Plot loss history
        train_losses, val_losses = zip(*loss_history)
        plt.figure()
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
        plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(output_dir, 'loss_history.png'))
        plt.close()

if __name__ == "__main__":

    train(params)