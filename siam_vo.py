import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

# from model_siam import SiamVO
from model_trans import ViTVO
from kitti_dataset import KITTIOdometryDataset

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import PIL.Image as Image

DISP_NORM = (0.95525, 0.43561)
YAW_NORM = (0.09490, 1.72319)

def twists2poses(twists):
    initial_pose = np.eye(4)
    poses = [initial_pose]
    current_pose = initial_pose.copy()

    for twist in twists:
        omega = twist[:3]
        v = twist[3:]

        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            R = np.eye(3)
        else:
            k = omega / theta
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        t = v

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        current_pose = current_pose @ transformation_matrix
        poses.append(current_pose.copy())

    return poses

def load_poses(pose_file):
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

def poses2yaw_displacement(poses):
    yaw_displacements = []
    for i in range(1, len(poses)):
        dx = poses[i][0, 3] - poses[i-1][0, 3]
        dz = poses[i][2, 3] - poses[i-1][2, 3]
        displacement = np.sqrt(dx**2 + dz**2)
        yaw = np.arctan2(dz, dx)
        yaw_displacements.append((displacement, yaw))
    return yaw_displacements

def plot_yaw_disp_traj(estm, gt):
    estm_x, estm_y = [0], [0]
    for disp, yaw in estm:
        x = estm_x[-1] + disp * np.cos(yaw)
        y = estm_y[-1] + disp * np.sin(yaw)
        estm_x.append(x)
        estm_y.append(y)

    gt_x, gt_y = [0], [0]
    for disp, yaw in gt:
        x = gt_x[-1] + disp * np.cos(yaw)
        y = gt_y[-1] + disp * np.sin(yaw)
        gt_x.append(x)
        gt_y.append(y)

    plt.figure()
    plt.plot(estm_x, estm_y, marker='o', label='Estimated')
    plt.plot(gt_x, gt_y, marker='x', label='Ground Truth')
    plt.title('Yaw and Displacement Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.axis('equal')
    plt.grid()    
    plt.legend()
    plt.tight_layout()
    plt.savefig('yaw_displacement_comparison.png')

def plot_poses(estm_poses, gt_poses):
    x_estm, z_estm = [], []
    for pose in estm_poses:
        x_estm.append(pose[0, 3])
        z_estm.append(pose[2, 3])

    x_gt, z_gt = [], []
    for pose in gt_poses:
        x_gt.append(pose[0, 3])
        z_gt.append(pose[2, 3])

    plt.figure()
    plt.plot(x_estm, z_estm, marker='o', label='Estimated')
    # plt.plot(x_gt, z_gt, marker='x', label='Ground Truth')
    plt.title('Camera Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig('trajectory_comparison.png')

def plot_yaw_disp_comp(estm, gt):
    estm_disps, estm_yaws = zip(*estm)
    gt_disps, gt_yaws = zip(*gt)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(estm_disps, label='Estimated')
    plt.plot(gt_disps, label='Ground Truth')
    plt.title('Displacement Comparison')
    plt.xlabel('Frame Index')
    plt.ylabel('Displacement (m)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(estm_yaws, label='Estimated')
    plt.plot(gt_yaws, label='Ground Truth')
    plt.title('Yaw Comparison')
    plt.xlabel('Frame Index')
    plt.ylabel('Yaw (radians)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('yaw_displacement_values_comparison.png')

def run_vo(dataset_root, pose_file, weights_path):

    transform = transforms.Compose([
        transforms.Resize(KITTIOdometryDataset.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3583], std=[0.3142])
    ])

    window_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTVO(KITTIOdometryDataset.IMAGE_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("Model loaded")

    images_dir = os.path.join(dataset_root, 'sequences', pose_file.removesuffix('.txt'), 'image_0')
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    # image_files = image_files[::6]  # Reduce rate to 5hz (get 1/6th of the dataset)

    yaw_disps = []
    for i in tqdm(range(len(image_files) - 2), desc="Running VO"):

        imgs = []
        for w in range(window_size):

            img_path = image_files[i + w]
            img = Image.open(img_path) 
            img = transform(img)
            img = img.unsqueeze(0) # add frames dimension
            imgs.append(img)

        # concat and convert to numpy
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs, dtype=np.float32)
        imgs = imgs.transpose(1, 0, 2, 3)  # (C, 2, H, W) -> (2, C, H, W)
        imgs = torch.tensor(imgs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, C, H, W) add batch dimension

        with torch.no_grad():
            y = model(imgs)

        disp, yaw = y.squeeze(0).cpu().numpy()

        # Denormalize
        disp = disp * DISP_NORM[1] + DISP_NORM[0]
        yaw = yaw * YAW_NORM[1] + YAW_NORM[0]

        yaw_disps.append((disp, yaw))
        # print(yaw_disp)

    print("Number of poses: ", len(yaw_disps))
    gt_poses = load_poses(os.path.join(dataset_root, 'poses', pose_file))
    gt_yaw_disp = poses2yaw_displacement(gt_poses)
    plot_yaw_disp_traj(yaw_disps, gt_yaw_disp)
    plot_yaw_disp_comp(yaw_disps, gt_yaw_disp)

if __name__ == "__main__":
    dataset_root = "/home/gopontarolo/siamese-vo/Datasets/kitti-odometry/dataset"
    test_seq = ['01', '03', '04', '05', '06', '07', '10']
    pose_file = "03.txt"
    weights_path = "checkpoints/2025-11-13_10-20-43/siamvo_epoch_88.pth"

    run_vo(dataset_root, pose_file, weights_path)
    # gt_poses = load_poses(os.path.join(dataset_root, 'poses', pose_file))
    # gt_yaw_disp = poses2yaw_displacement(gt_poses)
    # plot_yaw_disp(gt_yaw_disp)