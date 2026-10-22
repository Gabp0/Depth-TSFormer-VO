import torch
from tsformer.vit import VisionTransformer
from functools import partial
from torchvision import transforms
import numpy as np
import warnings

from kitti_dataset import KITTIOdometryDataset
from utils import euler_to_rotation

class ViTVO(torch.nn.Module):
    def __init__(self, params):
        super(ViTVO, self).__init__()
        self.params = params

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

        self.depth_pipeline = None

        print(f"Params num: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")

    def forward(self, x):
        x = self.backbone(x)
        return x

    def load_checkpoint(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.backbone.load_state_dict(state_dict)

    def infer_imgs(self, imgs, depths=None):

        transform = transforms.Compose([
            transforms.Resize(self.params["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[KITTIOdometryDataset.IMG_NORM[0]], std=[KITTIOdometryDataset.IMG_NORM[1]])
        ])

        x = []
        for i in range(len(imgs)):
            img = imgs[i]

            if depths is None:
                # initialize depth pipe only if required
                if self.depth_pipeline is None:
                    from transformers import pipeline
                    self.depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

                # supress huggingface warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    depth = self.depth_pipeline(img)['depth']

            else:
                depth = depths[i]

            img = transform(img)
            depth = transform(depth)
            img = torch.cat((img, depth), dim=0)  # concatenate image and depth channels
            img = img.unsqueeze(0) # add frames dimension
            x.append(img)

        x = np.concatenate(x, axis=0)
        x = np.asarray(x, dtype=np.float32)
        x = x.transpose(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)  # add batch dimension

        with torch.no_grad():
            y = self.forward(x)

        y = y.squeeze(0).cpu().numpy()

        # denormalize
        angles = y[:3] * KITTIOdometryDataset.ANGLE_NORM[1] + KITTIOdometryDataset.ANGLE_NORM[0]
        t = y[3:] * KITTIOdometryDataset.TRANSLATION_NORM[1] + KITTIOdometryDataset.TRANSLATION_NORM[0]

        z, y, x = angles
        if self.params["planar"]:
            z, x = 0, 0
            t[1] = 0

        # relative transform from the previous camera frame
        T_rel = np.eye(4)
        T_rel[:3, :3] = euler_to_rotation(z, y, x, seq='zyx')
        T_rel[:3, 3] = t

        return T_rel

if __name__ == "__main__":
    from params import params
    model = ViTVO(params)
