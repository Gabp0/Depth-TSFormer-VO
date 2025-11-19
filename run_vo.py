import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from model import ViTVO
from params import params

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import PIL.Image as Image

def yawdisp2poses(yaw_disps):
    poses = []
    curr_x, curr_y = 0.0, 0.0
    curr_yaw = 0.0
    for disp, yaw in yaw_disps:
        curr_yaw += yaw
        curr_x += disp * np.cos(yaw)
        curr_y += disp * np.sin(yaw)

        transformation_matrix = np.array([
            [np.cos(curr_yaw), -np.sin(curr_yaw), 0, curr_x],
            [0,                 0,                1, 0],
            [np.sin(curr_yaw),  np.cos(curr_yaw), 0, curr_y],
            [0,                 0,                0, 1]
        ])

        poses.append(transformation_matrix)
    return poses

def yawdisp2kitti_eval(yaw_disps):
    kitti_format = []
    poses = yawdisp2poses(yaw_disps)
    for pose in poses:
        flat_pose = pose[0:3, :].flatten()
        kitti_format.append(flat_pose.tolist())

    return kitti_format

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

def run_vo(dataset_root, seq, weights_path):

    window_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTVO(params).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("Model loaded")

    images_dir = os.path.join(dataset_root, 'sequences',seq, 'image_0')
    depth_dir = os.path.join(dataset_root, 'sequences',seq, 'depth_0')
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))

    yaw_disps = []
    for i in tqdm(range(len(image_files) - 2), desc="Running VO"):

        imgs = []
        depths = []
        for w in range(window_size):

            img = Image.open(image_files[i + w]) 
            depth = Image.open(depth_files[i + w])
            imgs.append(img)
            depths.append(depth)

        disp, yaw = model.infer_imgs(imgs)

        yaw_disps.append((disp, yaw))

    print("Number of poses: ", len(yaw_disps))

    pose_file = f"{seq}.txt"
    gt_poses = load_poses(os.path.join(dataset_root, 'poses', pose_file))
    gt_yaw_disp = poses2yaw_displacement(gt_poses)
    plot_yaw_disp_traj(yaw_disps, gt_yaw_disp)
    plot_yaw_disp_comp(yaw_disps, gt_yaw_disp)

    # save in KITTI format
    kitti_poses = yawdisp2kitti_eval(yaw_disps)
    output_file = f"{params['results_dir']}/{pose_file}"
    with open(output_file, 'w') as f:
        for pose in kitti_poses:
            f.write(' '.join(map(str, pose)) + '\n')
    print(f"Saved VO results to {output_file}")

if __name__ == "__main__":
    dataset_root = "Data/Datasets/kitti-odometry/dataset"
    weights_path = "Data/checkpoints/depth-tsformer_base-2025-11-13_23-46-55/dtsvo_epoch_77-best.pth"
    test_seq = ['05']

    for seq in test_seq:
        run_vo(dataset_root, seq, weights_path)