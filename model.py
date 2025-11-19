import torch
from tsformer.vit import VisionTransformer
from functools import partial

class ViTVO(torch.nn.Module):
    def __init__(self, params):
        super(ViTVO, self).__init__()

        self.backbone = VisionTransformer(img_size=params["img_size"], 
                                        patch_size=params["patch_size"],
                                        in_chans=params["in_chans"], 
                                        num_classes=params["num_classes"], 
                                        num_frames=params["num_frames"],
                                        embed_dim=params["embed_dim"], 
                                        depth=params["depth"], 
                                        num_heads=params["num_heads"], 
                                        mlp_ratio=params["mlp_ratio"], 
                                        qkv_bias=params["qkv_bias"],
                                        attn_drop_rate=params["attn_drop_rate"], 
                                        drop_path_rate=params["drop_path_rate"], 
                                        attention_type=params["attention_type"],
                                        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), 
                                        drop_rate=0.0) 

        print(self.backbone)
        print(f"Params num: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == "__main__":
    params = {
        # train params
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "val_split": 0.2,
        "dataset_root": "./Datasets/kitti-odometry/dataset",
        "output_dir": "./checkpoints",
        "img_size": (192, 640),
        "val_split": 0.2,

        # model params - small
        # "patch_size": 16,
        # "in_chans": 2,
        # "num_classes": 2,
        # "num_frames": 2,
        # "embed_dim": 384,
        # "depth": 12,
        # "num_heads": 6,
        # "mlp_ratio": 4,
        # "qkv_bias": True,
        # "attn_drop_rate": 0.1,
        # "drop_path_rate": 0.1,
        # "attention_type": 'divided_space_time'

        # model params - tiny
        "patch_size": 16,
        "in_chans": 2,
        "num_classes": 2,
        "num_frames": 2,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "attn_drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "attention_type": 'divided_space_time'
    }        

    model = ViTVO(params)
    # print model params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")