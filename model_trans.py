import torch
from models.vit import VisionTransformer
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

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == "__main__":
    model = ViTVO()
    # print model params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")