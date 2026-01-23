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

from scipy.signal import butter, filtfilt

world_T_cam = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

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

def poses2kitti(poses):
    kitti_format = []
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

            transformation_matrix = world_T_cam @ transformation_matrix

            poses.append(transformation_matrix)

    return poses

def low_pass_filter(poses, cutoff=0.1, fs=10.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    poses_array = np.array(poses)
    filtered_dx = filtfilt(b, a, poses_array[:, 0])
    filtered_dy = filtfilt(b, a, poses_array[:, 1])
    filtered_dyaw = filtfilt(b, a, poses_array[:, 2])

    filtered_poses = list(zip(filtered_dx, filtered_dy, filtered_dyaw))
    return filtered_poses


def sixdof2twodof(poses):
    two_dof = []
    for i in range(1, len(poses)):
        prev_pose = poses[i - 1]
        curr_pose = poses[i]
        rel_pose = curr_pose @ np.linalg.inv(prev_pose) 

        dx = rel_pose[0, 3]
        dy = rel_pose[1, 3]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        dyaw = np.arctan2(rel_pose[1, 0], rel_pose[0, 0])

        two_dof.append((r, theta, dyaw))

    return two_dof

def twodof2sixdof(twodof_poses):

    current_pose = np.eye(4)
    
    sixdof_poses = [current_pose.copy()]
    curr_x, curr_y, curr_yaw = 0, 0, 0
    for dx, dy, dyaw in twodof_poses:

        curr_yaw += dyaw
        curr_x += dx
        curr_y += dy

        new_pose = np.array([[np.cos(dyaw), -np.sin(dyaw), 0, curr_x],
                             [np.sin(dyaw),  np.cos(dyaw), 0, curr_y],
                             [0,             0,            1, 0],
                             [0,             0,            0, 1]])

        sixdof_poses.append(new_pose.copy())

    return sixdof_poses

def plot_6dof_traj(estm, gt):
    estm_x = [p[0, 3] for p in estm]
    estm_y = [p[1, 3] for p in estm]
    estm_yaw = [np.arctan2(p[1, 0], p[0, 0]) for p in estm]

    gt_x = [p[0, 3] for p in gt]
    gt_y = [p[1, 3] for p in gt]
    gt_yaw = [np.arctan2(p[1, 0], p[0, 0]) for p in gt]

    plt.figure()
    plt.plot(estm_x, estm_y, marker='o', label='Estimated')
    plt.plot(gt_x, gt_y, marker='x', label='Ground Truth')
    plt.title('2D Pose Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.axis('equal')
    plt.grid()    
    plt.legend()
    plt.tight_layout()
    plt.savefig('2d_trajectory.png')

    plt.clf()
    plt.plot(estm_yaw, marker='o', label='Estimated')
    plt.plot(gt_yaw, marker='x', label='Ground Truth')
    plt.title('Yaw Angle over Time')
    plt.xlabel('Frame')
    plt.ylabel('Yaw (rad)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('yaw_over_time.png')

def plot_components(twodof_poses, gt_2dof_poses):
    estm_dx = [p[0] for p in twodof_poses]
    estm_dy = [p[1] for p in twodof_poses]
    estm_dyaw = [p[2] for p in twodof_poses]

    gt_dx = [p[0] for p in gt_2dof_poses]
    gt_dy = [p[1] for p in gt_2dof_poses]
    gt_dyaw = [p[2] for p in gt_2dof_poses]

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(estm_dx, label='Estimated dx', marker='o')
    plt.plot(gt_dx, label='Ground Truth dx', marker='x')
    plt.title('Displacement in X over Time')
    plt.xlabel('Frame')
    plt.ylabel('dx (m)')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(estm_dy, label='Estimated dy', marker='o')
    plt.plot(gt_dy, label='Ground Truth dy', marker='x')
    plt.title('Displacement in Y over Time')
    plt.xlabel('Frame')
    plt.ylabel('dy (m)')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(estm_dyaw, label='Estimated dyaw', marker='o')
    plt.plot(gt_dyaw, label='Ground Truth dyaw', marker='x')
    plt.title('Yaw Change over Time')
    plt.xlabel('Frame')
    plt.ylabel('dyaw (rad)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig('pose_components.png')

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

    twodof_poses = []
    for i in tqdm(range(len(image_files) - 2), desc="Running VO"):

        imgs = []
        depths = []
        for w in range(window_size):

            img = Image.open(image_files[i + w]) 
            depth = Image.open(depth_files[i + w])
            imgs.append(img)
            depths.append(depth)

        dx, dy, dyaw = model.infer_imgs(imgs, depths)

        twodof_poses.append((dx, dy, dyaw))

    print("Number of poses: ", len(twodof_poses))

    # twodof_poses = low_pass_filter(twodof_poses, cutoff=0.1, fs=10.0)
    sixdof_poses = twodof2sixdof(twodof_poses)
    
    gt_poses = load_poses(os.path.join(dataset_root, 'poses', f"{seq}.txt"))
    gt_2dof_poses = sixdof2twodof(gt_poses)
    gt_poses_reduced = twodof2sixdof(gt_2dof_poses)

    plot_6dof_traj(sixdof_poses, gt_poses)
    plot_components(twodof_poses, gt_2dof_poses)

    # save in KITTI format
    kitti_poses = poses2kitti(sixdof_poses)
    output_file = f"{params['results_dir']}/{seq}.txt"
    with open(output_file, 'w') as f:
        for pose in kitti_poses:
            f.write(' '.join(map(str, pose)) + '\n')
    print(f"Saved VO results to {output_file}")

if __name__ == "__main__":
    dataset_root = "Data/Datasets/kitti-odometry/dataset"
    weights_path = "Data/checkpoints/2026-01-21_21-37-39/dtsvo_epoch_73-best.pth"
    test_seq = ['01', '03', '04', '05', '06', '07', '10']

    for seq in test_seq:
        run_vo(dataset_root, seq, weights_path)