from kitti_dataset import KITTIOdometryDataset

params = {
    # train params
    "batch_size": 8,
    "epochs": 100,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "dataset_root": "./Data/Datasets/kitti-odometry/dataset",
    "output_dir": "./Data/checkpoints",
    "val_split": 0.2,
    "img_size": KITTIOdometryDataset.IMAGE_SIZE,
    "disp_norm": KITTIOdometryDataset.DISP_NORM,
    "yaw_norm": KITTIOdometryDataset.YAW_NORM,

    # depth params
    "depth_size": 'vits',
    "depth_weights": './Data/checkpoints/depth-anything-small/depth_anything_v2_vits.pth',  # path to depth model weights

    "patch_size": 16,
    "in_chans": 2,
    "num_classes": 2,
    "num_frames": 2,
    "depth": 12,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "attn_drop_rate": 0.1,
    "drop_path_rate": 0.1,
    "attention_type": 'divided_space_time',

    # model params - base
    "results_dir": "./Data/results/dtsvo-base",
    "embed_dim": 384,
    "num_heads": 6

    # model params - small
    # "results_dir": "./Data/results/dtsvo-small",
    # "embed_dim": 192,
    # "num_heads": 3
}        
