import os
import glob
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation
from time import sleep

class KITTIOdometryDataset(Dataset):

    IMAGE_SIZE = (192, 640)
    DISP_NORM = (0.95525, 0.43561)
    YAW_NORM = (0.09490, 1.72319)

    def __init__(self, window_size, root_dir, sequence_numbers, transform = None):
        self.window_size = window_size
        self.root_dir = root_dir
        self.sequence_numbers = sequence_numbers
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3583], std=[0.3142])
            ])
        
        self.dataset_sequences = []
        self.dataset_depths = []
        self.dataset_poses = []
        self.total_length = 0
        for sequence_num in sequence_numbers:
            image_dir = os.path.join(root_dir, 'sequences', sequence_num, 'image_0')
            depth_dir = os.path.join(root_dir, 'sequences', sequence_num, 'depth_0')
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
            pose_file = os.path.join(root_dir, 'poses', sequence_num + ".txt")
            poses = self.load_poses(pose_file)
            self.dataset_sequences.append(image_files)
            self.dataset_depths.append(depth_files)
            self.dataset_poses.append(poses)
            self.total_length += len(image_files) - (self.window_size - 1)
            print(f"Loaded sequence {sequence_num} with {len(image_files)} images")
        print(f"Total dataset length (window size {self.window_size}): {self.total_length} samples")

        # values calculated for the complete dataset
        self.disp_norm = self.DISP_NORM  
        self.yaw_norm =  self.YAW_NORM   

        print(f"Dataset normalization values: ")
        print(f"Disparity - mean: {self.disp_norm[0]}  std dev: {self.disp_norm[1]}")
        print(f"Yaw       - mean: {self.yaw_norm[0]}  std dev: {self.yaw_norm[1]}")

        # disp: mean 0.9552555071464498  std dev 0.43561040902403037
        # yaw:  mean 0.09490150096396582 std dev 1.7231921715542085

        # Reduce rate to 5hz (get 1/6th of the dataset)
        # self.image_files = self.image_files[::6]
        # self.poses = self.poses[::6]

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

    def pose2yaw_disp(self, prev_pose, curr_pose):

        # get 2d trajectory
        dx = curr_pose[0, 3] - prev_pose[0, 3]
        dz = curr_pose[2, 3] - prev_pose[2, 3]

        displacement = np.sqrt(dx**2 + dz**2)
        yaw = np.arctan2(dz, dx)
        return displacement, yaw
    
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # get sequence and local index
        seq_idx = 0
        while idx >= len(self.dataset_sequences[seq_idx]) - (self.window_size - 1):
            idx -= (len(self.dataset_sequences[seq_idx]) - (self.window_size - 1))
            seq_idx += 1

        image_files = self.dataset_sequences[seq_idx]
        depth_files = self.dataset_depths[seq_idx]
        poses = self.dataset_poses[seq_idx]    

        # load images and apply transforms
        imgs = []
        for w in range(self.window_size):

            if idx + w >= len(image_files):
                print(f"Index {idx} {w} out of bounds for sequence length {len(image_files)}")

            img = Image.open(image_files[idx + w]) 
            depth = Image.open(depth_files[idx + w])
            if self.transform:
                img = self.transform(img)
                depth = self.transform(depth)

            img = torch.cat((img, depth), dim=0) # concatenate image and depth as different channels
            img = img.unsqueeze(0) # add new frame dimension

            imgs.append(img)
        
        # concat and convert to numpy
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs, dtype=np.float32)
        imgs = imgs.transpose(1, 0, 2, 3)  # (C, 2, H, W) -> (2, C, H, W)

        # get first and last gt pose 
        first_pose = poses[idx]
        last_pose = poses[idx + (self.window_size - 1)]
        disp, yaw = self.pose2yaw_disp(first_pose, last_pose)

        # normalize
        disp = (disp - self.disp_norm[0]) / self.disp_norm[1]
        yaw = (yaw - self.yaw_norm[0]) / self.yaw_norm[1]
        gt = torch.tensor([disp, yaw], dtype=torch.float32)

        return (imgs, gt)

def plot_yaw_disp(gt):
    x = [0]
    y = [0]
    for disp, yaw in gt:
        disp = disp * KITTIOdometryDataset.DISP_NORM[1] + KITTIOdometryDataset.DISP_NORM[0]
        yaw = yaw * KITTIOdometryDataset.YAW_NORM[1] + KITTIOdometryDataset.YAW_NORM[0]
        x = x + [x[-1] + disp * np.cos(yaw)]
        y = y + [y[-1] + disp * np.sin(yaw)]

    plt.figure()
    plt.plot(x, y, marker='o', label='Ground Truth')
    plt.title('Yaw and Displacement Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.axis('equal')
    plt.grid()    
    plt.legend()
    plt.tight_layout()
    plt.savefig('yaw_displacement_comparison.png')

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    val_split = 0.2
    dataset = KITTIOdometryDataset(3, "./Datasets/kitti-odometry/dataset", ["00", "01"])

    val_len = int(len(dataset) * val_split)
    test_len = int(len(dataset) - val_len)
    train_data, val_data = torch.utils.data.random_split(dataset, [test_len, val_len], generator=torch.Generator().manual_seed(99))
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

    poses = []
    first_img = True
    for imgs, gt in train_loader:
        if first_img:
            print(imgs.shape)
            first_img = False

        disp, yaw = gt.squeeze(0).cpu().numpy()
        poses.append((disp, yaw))

    for imgs, gt in val_loader:
        disp, yaw = gt.squeeze(0).cpu().numpy()
        poses.append((disp, yaw))

    plot_yaw_disp(poses)