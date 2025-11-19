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

euler_angles = []
delta_pos = []

class KITTIOdometryDataset(Dataset):

    IMAGE_SIZE = (192, 640)
    DISP_NORM = (0.95525, 0.43561)
    YAW_NORM = (0.09490, 1.72319)

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

                transformation_matrix = self.world_T_cam @ transformation_matrix
                poses.append(transformation_matrix)
                
        return poses

    def pose2yaw_disp(self, prev_pose, curr_pose):

        rel_pose = curr_pose @ np.linalg.inv(prev_pose)
        position = rel_pose[0:3, 3]# @ rel_pose[0:3, 0:3]
        dx = position[0]
        dy = position[1]
        dz = position[2]

        dx2 = curr_pose[0, 3] - prev_pose[0, 3]
        dy2 = curr_pose[1, 3] - prev_pose[1, 3]
        yaw_pos = np.arctan2(dx2, dy2)
        yaw_pos2 = np.arctan2(dx, dy)

        angles = Rotation.from_matrix(rel_pose[0:3, 0:3]).as_euler('xyz', degrees=False)
        self.euler_angles.append([angles[2], yaw_pos, yaw_pos2])
        self.delta_pos.append([dx, dy, dz])
        if len(self.euler_angles) % 50 == 0:
            plot_euler_angles(self.euler_angles, self.delta_pos)

        # dx2 = curr_pose[0, 3] - prev_pose[0, 3]
        # dy2 = curr_pose[1, 3] - prev_pose[1, 3]
        # dz2 = curr_pose[2, 3] - prev_pose[2, 3]

        # print(f"Delta X: {dx} vs {dx2}, Delta Y: {dy} vs {dy2}, Delta Z: {dz} vs {dz2}")
                                    
        displacement = np.sqrt(dx**2 + dy**2)
        yaw = np.arctan2(rel_pose[1, 0], rel_pose[0, 0])

        yaw_pos = np.arctan2(dx, dy)
        # print(f"Yaw pos: {yaw_pos}, Yaw change: {yaw}")

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
        # for w in range(self.window_size):

        #     if idx + w >= len(image_files):
        #         print(f"Index {idx} {w} out of bounds for sequence length {len(image_files)}")

        #     img = Image.open(image_files[idx + w]) 
        #     depth = Image.open(depth_files[idx + w])
        #     if self.transform:
        #         img = self.transform(img)
        #         depth = self.transform(depth)

        #     img = torch.cat((img, depth), dim=0) # concatenate image and depth as different channels
        #     img = img.unsqueeze(0) # add new frame dimension

        #     imgs.append(img)
        
        # # concat and convert to numpy
        # imgs = np.concatenate(imgs, axis=0)
        # imgs = np.asarray(imgs, dtype=np.float32)
        # imgs = imgs.transpose(1, 0, 2, 3)  # (C, 2, H, W) -> (2, C, H, W)

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
    current_pose = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    poses = [current_pose]
    for disp, yaw in gt:
        disp = disp * KITTIOdometryDataset.DISP_NORM[1] + KITTIOdometryDataset.DISP_NORM[0]
        yaw = yaw * KITTIOdometryDataset.YAW_NORM[1] + KITTIOdometryDataset.YAW_NORM[0]
        
        rel_pose = np.array([[np.cos(yaw), -np.sin(yaw), 0, disp * np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw), 0, disp * np.cos(yaw)],
                            [0,            0,           1, 0],
                            [0,            0,           0, 1]])
        
        current_pose = current_pose @ rel_pose
        poses.append(current_pose)

    x = []
    y = []
    for pose in poses:
        x.append(pose[0, 3])
        y.append(pose[1, 3])

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

    val_split = 0.0
    dataset = KITTIOdometryDataset(3, "./Data/Datasets/kitti-odometry/dataset", ["00"])

    val_len = int(len(dataset) * val_split)
    test_len = int(len(dataset) - val_len)
    # train_data, val_data = torch.utils.data.random_split(dataset, [test_len, val_len], generator=torch.Generator().manual_seed(99))
    
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    poses = []
    first_img = True
    for imgs, gt in train_loader:
        if first_img:
            first_img = False

        disp, yaw = gt.squeeze(0).cpu().numpy()
        poses.append((disp, yaw))

    # for imgs, gt in val_loader:
    #     disp, yaw = gt.squeeze(0).cpu().numpy()
    #     poses.append((disp, yaw))

    plot_yaw_disp(poses)