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

class KITTIOdometryDataset(Dataset):
    def __init__(self, root_dir, poses_file, transform = None, target_transform = None):
        self.root_dir = root_dir
        self.poses_file = poses_file
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_dir = os.path.join(root_dir, 'sequences', poses_file.removesuffix('.txt'), 'image_0')
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.pose_file = os.path.join(root_dir, 'poses', poses_file)
        self.poses = self.load_poses(self.pose_file)

        # Reduce rate to 5hz (get 1/6th of the dataset)
        self.image_files = self.image_files[::6]
        self.poses = self.poses[::6]

    def load_poses(self, pose_file):
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                # r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
                pose = list(map(float, line.strip().split()))
                transformation_matrix = np.array([
                    [pose[0], pose[1], pose[2], pose[3]],
                    [pose[4], pose[5], pose[6], pose[7]],
                    [pose[8], pose[9], pose[10], pose[11]],
                    [0, 0, 0, 1]
                ])
                poses.append(transformation_matrix)

        return poses
    
    def __len__(self):
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        # get image pair
        img1_path = self.image_files[idx]
        img2_path = self.image_files[idx + 1]

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # get gt poses 
        pose1 = self.poses[idx]
        pose2 = self.poses[idx + 1]

        # compute twist
        pose_rel = np.linalg.inv(pose1) @ pose2
        tx = pose_rel[0, 3]
        ty = pose_rel[1, 3]
        tz = pose_rel[2, 3]
        rx = np.arctan2(pose_rel[2, 1], pose_rel[2, 2])
        ry = np.arctan2(-pose_rel[2, 0], np.sqrt(pose_rel[2, 1]**2 + pose_rel[2, 2]**2))
        rz = np.arctan2(pose_rel[1, 0], pose_rel[0, 0])
        twist = torch.tensor([tx, ty, tz, rx, ry, rz], dtype=torch.float32)

        return (img1, img2, twist)
    
class RelativePoseLoss(torch.nn.Module):
    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super(RelativePoseLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(self, pred, target):
        translation_loss = F.mse_loss(pred[:, :3], target[:, :3])
        rotation_loss = F.mse_loss(pred[:, 3:], target[:, 3:])
        total_loss = (self.translation_weight * translation_loss +
                      self.rotation_weight * rotation_loss)
        return total_loss