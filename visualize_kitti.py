import numpy as np
import matplotlib.pyplot as plt

import os
import glob

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

    # get 2d trajectory
    x = []
    z = []
    for pose in poses:
        x.append(pose[0, 3])
        z.append(pose[2, 3])

    yaws = []
    displacements = []
    for i in range(1, len(poses)):
        dx = x[i] - x[i - 1]
        dz = z[i] - z[i - 1]
        displacement = np.sqrt(dx**2 + dz**2)
        yaw = np.arctan2(dz, dx)
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
    x, z = [], []
    for pose in poses:
        x.append(pose[0, 3])
        z.append(pose[2, 3])

    plt.figure()
    plt.plot(x, z, marker='o', label='Ground Truth')
    plt.title('Camera Trajectory')
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig('trajectory_comparison.png')

if __name__ == "__main__":
    dataset_root = "/home/gopontarolo/siamese-vo/Datasets/kitti-odometry/dataset"
    pose_file = "00.txt"

    gt_poses = load_poses(os.path.join(dataset_root, 'poses', pose_file))
    gt_yaw_disp = poses2yaw_displacement(gt_poses)
    plot_yaw_disp(gt_yaw_disp)
    plot_poses(gt_poses)