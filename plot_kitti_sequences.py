import numpy as np
import matplotlib.pyplot as plt
import os

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


if __name__ == "__main__":
    sequences = ['03', '04', '05', '06', '07', '10']
    gt_path = "Data/Datasets/kitti-odometry/dataset/poses/"
    res_path = "Data/results/dtsvo-base/"

    # plot trajectories and yaw/displacement comparison side by side for each sequence
    
    rows = len(sequences)
    fig, axs = plt.subplots(rows, 3, figsize=(20, 5 * rows))

    for i, seq in enumerate(sequences):
        gt_poses = load_poses(os.path.join(gt_path, f"{seq}.txt"))
        estm_poses = load_poses(os.path.join(res_path, f"{seq}.txt"))

        gt_yaw_disp = poses2yaw_displacement(gt_poses)
        estm_yaw_disp = poses2yaw_displacement(estm_poses)

        # Plot trajectories
        ax_traj = axs[i, 0]
        gt_x, gt_y = [0], [0]
        for disp, yaw in gt_yaw_disp:
            x = gt_x[-1] + disp * np.cos(yaw)
            y = gt_y[-1] + disp * np.sin(yaw)
            gt_x.append(x)
            gt_y.append(y)
        ax_traj.plot(gt_x, gt_y, marker='o', label='Ground Truth')

        estm_x, estm_y = [0], [0]
        for disp, yaw in estm_yaw_disp:
            x = estm_x[-1] + disp * np.cos(yaw)
            y = estm_y[-1] + disp * np.sin(yaw)
            estm_x.append(x)
            estm_y.append(y)
        ax_traj.plot(estm_x, estm_y, marker='x', label='Estimated')

        ax_traj.set_xlabel('X position (m)')
        ax_traj.set_ylabel('Z position (m)')
        ax_traj.axis('equal')
        ax_traj.grid()
        ax_traj.legend()

        ax_comp = axs[i, 1]
        ax_comp2 = axs[i, 2]

        gt_disps, gt_yaws = zip(*gt_yaw_disp)
        estm_disps, estm_yaws = zip(*estm_yaw_disp)

        ax_comp.plot(gt_disps, label='GT Displacement')
        ax_comp.plot(estm_disps, label='Est. Displacement')
        ax_comp.set_xlabel('Frame')
        ax_comp.set_ylabel('Displacement (m)')
        ax_comp.grid()
        ax_comp.legend()

        ax_comp2.plot(gt_yaws, label='GT Yaw')
        ax_comp2.plot(estm_yaws, label='Est. Yaw')
        ax_comp2.set_xlabel('Frame')
        ax_comp2.set_ylabel('Yaw (rad)')
        ax_comp2.grid()
        ax_comp2.legend()
    
    plt.tight_layout()

    # set overall title
    fig.suptitle('Trajectory, Displacement and Yaw comparison for the test sequences', fontsize=18)

    # adjust layout to make room for the suptitle
    plt.subplots_adjust(top=0.95)

    plt.savefig('kitti_sequences_comparison.png')