import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from scipy.spatial.transform import Rotation as R

cam_T_world = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

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
            transformation_matrix = cam_T_world @ transformation_matrix

            poses.append(transformation_matrix)

    return poses

def poses2yaw_displacement(poses):

    # get 2d trajectory
    x = []
    y = []
    for pose in poses:
        x.append(pose[0, 3])
        y.append(pose[2, 3])

    yaws = []
    displacements = []
    for i in range(1, len(poses)):
        rel_pose = np.linalg.inv(poses[i-1]) @ poses[i]
        dx = rel_pose[0, 3]
        dy = rel_pose[1, 3]
        yaw = R.from_matrix(rel_pose[0:3, 0:3]).as_euler('xyz', degrees=False)[0]
        displacement = np.sqrt(dx**2 + dy**2)

        displacements.append(displacement)
        yaws.append(yaw)

    return list(zip(displacements, yaws))

def plot_yaw_disp(gt):
    x = [0]
    y = [0]
    for disp, yaw in gt:
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

def plot_poses(poses):
    x, y = [], []
    for pose in poses:
        x.append(pose[0, 3])
        y.append(pose[1, 3])

    plt.figure()
    plt.plot(x, y, marker='o', label='Ground Truth')
    plt.title('Camera Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig('trajectory_comparison.png')

def plot_euler_angles(euler_angles):
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    euler_angles = np.array(euler_angles)
    ax[0].plot(euler_angles[:, 0], label='Roll')
    ax[0].set_title('Roll over Time')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Angle (rad)')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(euler_angles[:, 1], label='Pitch', color='orange')
    ax[1].set_title('Pitch over Time')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Angle (rad)')
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(euler_angles[:, 2], label='Yaw', color='green')
    ax[2].set_title('Yaw over Time')
    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('Angle (rad)')
    ax[2].grid()
    ax[2].legend()
    plt.tight_layout()
    plt.savefig('euler_angles.png')

def get_euler_angles_from_poses(poses):

    euler_angles = []
    for pose in poses:
        rotation_matrix = pose[0:3, 0:3]
        r = R.from_matrix(rotation_matrix)
        euler = r.as_euler('xyz', degrees=False)
        euler_angles.append(euler)

    return np.array(euler_angles)

if __name__ == "__main__":
    dataset_root = "./Data/Datasets/kitti-odometry/dataset"
    pose_file = "00.txt"

    gt_poses = load_poses(os.path.join(dataset_root, 'poses', pose_file))
    gt_yaw_disp = poses2yaw_displacement(gt_poses)
    euler_angles = get_euler_angles_from_poses(gt_poses)
    plot_yaw_disp(gt_yaw_disp)
    plot_poses(gt_poses)
    plot_euler_angles(euler_angles)