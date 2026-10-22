import os
import numpy as np
import matplotlib.pyplot as plt


def load_kitti_poses(file_path):
    """Load KITTI pose file into Nx3 positions (x, y, z)."""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            T = np.array(values).reshape(3, 4)
            position = T[:, 3]
            poses.append(position)
    return np.array(poses)  # shape (N, 3)


def plot_trajectory(ax, traj, label, color, use_y=False, min_length=0):
    """Plot trajectory: x vs z (default) or x vs y (if use_y=True)."""
    if use_y:
        ax.plot(-traj[:, 1], traj[:, 0], label=label, color=color, linewidth=1.5)
    else:
        ax.plot(traj[:, 0], traj[:, 2], label=label, color=color, linewidth=1.5)

    if min_length > 0:
        ax.set_xlim(-min_length, min_length)


def main():
    # ==== CONFIG ====
    sequences = ["10"]

    gt_dir = "/home/gopontarolo/depth-tsformer-vo/kitti-odom-eval/dataset/kitti_odom/gt_poses"
    method_dirs = [
        "/home/gopontarolo/TSformer-VO/checkpoints/exp_dtsvo-v3.1-small-lambda01/checkpoint_best/pred_poses",
        "/home/gopontarolo/depth-tsformer-vo/Data/results/dtsvo-v2_small"
    ]
    method_names = ["DTSVO-λ", "DTSVO-s"]

    colors = ["blue", "green"]

    # ==== PLOTTING ====
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 5 rows, 3 columns

    for col, (method_dir, method_name, color) in enumerate(zip(method_dirs, method_names, colors)):
        for row, seq in enumerate(sequences):
            ax = axes[row, col]

            gt_file = os.path.join(gt_dir, f"{seq}.txt")
            pred_file = os.path.join(method_dir, f"{seq}.txt")

            if not os.path.exists(gt_file) or not os.path.exists(pred_file):
                ax.set_title(f"{seq} (missing)")
                ax.axis("off")
                continue

            gt_traj = load_kitti_poses(gt_file)
            pred_traj = load_kitti_poses(pred_file)

            # Method 3 uses Y axis instead of Z
            use_y = (col == 1)

            # Sequence 04 needs a larger X range
            min_length = 100 if seq == "04" else 0

            # Plot GT and prediction
            plot_trajectory(ax, gt_traj, "GT", "black", use_y=False, min_length=min_length)
            plot_trajectory(ax, pred_traj, method_name, color, use_y=use_y, min_length=min_length)

            ax.set_title(f"Seq {seq}")
            ax.set_aspect('equal')
            ax.grid(True)

            # Labels
            if col == 0:
                ax.set_ylabel(f"Seq {seq}")
            if row == 0:
                ax.set_title(method_name)

            # Legend only once per column (top row)
            if row == 0:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("kitti_trajectories.png", dpi=300)


if __name__ == "__main__":
    main()