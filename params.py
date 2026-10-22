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
    "results_dir": "./Data/results/dtsvo-v3.1-small",
    "img_size": KITTIOdometryDataset.IMAGE_SIZE,

    # inference params
    "planar": False,  # keep only yaw and in-plane translation when recovering poses

    # model params - small 
    "patch_size": 16,
    "in_chans": 2,
    "num_classes": 6,
    "num_frames": 2,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "attn_drop_rate": 0.1,
    "drop_path_rate": 0.1,
    "attention_type": 'divided_space_time',
}
