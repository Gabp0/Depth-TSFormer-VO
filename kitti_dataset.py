import os
import glob
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import rotation_to_euler

class KITTIOdometryDataset(Dataset):

    IMAGE_SIZE = (192, 640)

    IMG_NORM = (0.34721234, 0.30737526)
    ANGLE_NORM = (np.array([1.7061e-5, 9.5582e-4, -5.5258e-5]), np.array([2.8256e-3, 1.7771e-2, 3.2326e-3]))
    TRANSLATION_NORM = (np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1]), np.array([2.5584e-2, 1.8545e-2, 3.0352e-1]))

    def __init__(self, window_size, root_dir, sequence_numbers, transform = None):
        self.window_size = window_size
        self.root_dir = root_dir
        self.sequence_numbers = sequence_numbers
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.IMG_NORM[0]], std=[self.IMG_NORM[1]])
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

            self.total_length += np.floor(len(image_files) / self.window_size).astype(int)

            print(f"Loaded sequence {sequence_num} with {len(image_files)} images")
        print(f"Total dataset length (window size {self.window_size}): {self.total_length} samples")

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

    def get_six_dof_pose(self, prev_pose, curr_pose):
        pose_wrt_prev = np.dot(np.linalg.inv(prev_pose), curr_pose)
        R = pose_wrt_prev[:3, :3]
        angles = rotation_to_euler(R, seq='zyx')
        translation = pose_wrt_prev[:3, 3]

        return np.array(angles), np.array(translation)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # get sequence and local index
        seq_idx = 0
        while idx >= np.floor(len(self.dataset_sequences[seq_idx]) / self.window_size).astype(int):
            idx -= np.floor(len(self.dataset_sequences[seq_idx]) / self.window_size).astype(int)
            seq_idx += 1

        image_files = self.dataset_sequences[seq_idx]
        depth_files = self.dataset_depths[seq_idx]
        poses = self.dataset_poses[seq_idx]

        # load images and apply transforms
        imgs = []
        idx = idx * self.window_size
        for w in range(self.window_size):
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
        imgs = imgs.transpose(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

        # relative pose between first and last frame, in the camera frame
        first_pose = poses[idx]
        last_pose = poses[idx + (self.window_size - 1)]
        angles, translation = self.get_six_dof_pose(first_pose, last_pose)

        # normalize
        angles = (angles - self.ANGLE_NORM[0]) / self.ANGLE_NORM[1]
        translation = (translation - self.TRANSLATION_NORM[0]) / self.TRANSLATION_NORM[1]

        gt = torch.tensor(np.concatenate((angles, translation)), dtype=torch.float32)

        return (imgs, gt)
