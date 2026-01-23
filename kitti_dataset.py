import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

euler_angles = []
delta_pos = []

class KITTIOdometryDataset(Dataset):

    IMAGE_SIZE = (192, 640)
    R_NORM = (0.95525, 0.43561)
    THETA_NORM = (0.09490, 1.72319)
    DYAW_NORM = (-0.000440879, 0.01727572)

    world_T_cam = np.array([[0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]])

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
        
        self.euler_angles = []
        self.delta_pos = []

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

        # values calculated for the complete dataset
        self.r_norm = self.R_NORM
        self.theta_norm = self.THETA_NORM
        self.dyaw_norm = self.DYAW_NORM

        # all_r, all_theta, all_dyaw = [], [], []
        # for seq_idx in range(len(self.dataset_poses)):
        #     poses = self.dataset_poses[seq_idx]
        #     for i in range(len(poses) - 1):
        #         r, theta, dyaw = self.six_dof2three_dof(poses[i], poses[i + 1])
        #         all_r.append(r)
        #         all_theta.append(theta)
        #         all_dyaw.append(dyaw)

        # self.r_norm = (np.mean(all_r), np.std(all_r))
        # self.theta_norm = (np.mean(all_theta), np.std(all_theta))
        # self.dyaw_norm = (np.mean(all_dyaw), np.std(all_dyaw))

        print(f"Dataset normalization values: ")
        print(f"  r norm: mean {self.r_norm[0]}, std {self.r_norm[1]}")
        print(f"  theta norm: mean {self.theta_norm[0]}, std {self.theta_norm[1]}")
        print(f"  dyaw norm: mean {self.dyaw_norm[0]}, std {self.dyaw_norm[1]}")

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

                transformation_matrix = self.world_T_cam @ transformation_matrix
                poses.append(transformation_matrix)
                
        return poses

    def six_dof2three_dof(self, prev_pose, curr_pose):

        prev_yaw = np.arctan2(prev_pose[1, 0], prev_pose[0, 0])
        curr_yaw = np.arctan2(curr_pose[1, 0], curr_pose[0, 0])
        dyaw = curr_yaw - prev_yaw

        dx = curr_pose[0, 3] - prev_pose[0, 3]
        dy = curr_pose[1, 3] - prev_pose[1, 3]

        return dx, dy, dyaw
    
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
        dx, dy, dyaw = self.six_dof2three_dof(first_pose, last_pose)

        # normalize

        gt = torch.tensor([dx, dy, dyaw], dtype=torch.float32)

        return (imgs, gt)

def plot_2dof_pose(gt):

    current_pose = np.eye(4)
    
    poses = [current_pose.copy()]
    curr_x, curr_y, curr_yaw = 0, 0, 0
    for dx, dy, dyaw in gt:

        curr_yaw += dyaw
        curr_x += dx
        curr_y += dy

        new_pose = np.array([[np.cos(dyaw), -np.sin(dyaw), 0, curr_x],
                             [np.sin(dyaw),  np.cos(dyaw), 0, curr_y],
                             [0,             0,            1, 0],
                             [0,             0,            0, 1]])

        # current_pose = new_pose @ current_pose
        poses.append(new_pose.copy())

    x = [p[0, 3] for p in poses]
    y = [p[1, 3] for p in poses]
    yaw = [np.arctan2(p[1, 0], p[0, 0]) for p in poses]

    plt.figure()
    plt.plot(x, y, marker='o', label='Ground Truth')
    plt.title('2D Pose Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.axis('equal')
    plt.grid()    
    plt.legend()
    plt.tight_layout()
    plt.savefig('2d_trajectory.png')

    plt.clf()
    plt.plot(yaw, marker='o', label='Yaw Angle')
    plt.title('Yaw Angle over Time')
    plt.xlabel('Frame')
    plt.ylabel('Yaw (rad)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('yaw_over_time.png')

def plot_euler_angles(euler_angles, delta_pos):
    euler_angles = np.array(euler_angles)
    print(euler_angles.shape)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(euler_angles[:, 0], label='Delta Pitch')
    axs[0].set_title('Delta Pitch over Time')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Delta Pitch (rad)')
    axs[0].grid()
    axs[1].plot(euler_angles[:, 1], label='Delta Roll', color='orange')
    axs[1].set_title('Delta Roll over Time')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Delta Roll (rad)')
    axs[1].grid()
    axs[2].plot(euler_angles[:, 2], label='Delta Yaw', color='green')
    axs[2].set_title('Delta Yaw over Time')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Delta Yaw (rad)')
    axs[2].grid()
    plt.tight_layout()
    plt.savefig('euler_angles.png')

    plt.clf()
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    delta_pos_arr = np.array(delta_pos)
    axs[0].plot(delta_pos_arr[:, 0], label='Delta X')
    axs[0].set_title('Delta X over Time')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Delta X (m)')
    axs[0].grid()
    axs[1].plot(delta_pos_arr[:, 1], label='Delta Y', color='orange')
    axs[1].set_title('Delta Y over Time')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Delta Y (m)')
    axs[1].grid()
    axs[2].plot(delta_pos_arr[:, 2], label='Delta Z', color='green')
    axs[2].set_title('Delta Z over Time')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Delta Z (m)')
    axs[2].grid()
    plt.tight_layout()
    plt.savefig('delta_positions.png')
    plt.close('all')

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = KITTIOdometryDataset(2, "./Data/Datasets/kitti-odometry/dataset", ['00'])
    # dataset = KITTIOdometryDataset(2, "./Data/Datasets/kitti-odometry/dataset", ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    poses = []
    first_img = True
    for imgs, gt in tqdm(train_loader):
        if first_img:
            first_img = False

        dx, dy, dyaw = gt[0].numpy()
        poses.append((dx, dy, dyaw))

    plot_2dof_pose(poses)