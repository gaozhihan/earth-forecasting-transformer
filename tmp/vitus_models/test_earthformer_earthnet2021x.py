import os
from omegaconf import OmegaConf
import torch
from earthformer.config import cfg
from earthformer.cuboid_transformer.cuboid_transformer_earthnet2021x import CuboidTransformerModelEarthNet2021x


def config_cuboid_transformer(cfg):
    model_cfg = OmegaConf.to_object(cfg.model)
    num_blocks = len(model_cfg["enc_depth"])
    if isinstance(model_cfg["self_pattern"], str):
        enc_attn_patterns = [model_cfg.pop("self_pattern")] * num_blocks
    else:
        enc_attn_patterns = OmegaConf.to_container(model_cfg.pop("self_pattern"))
    model_cfg["enc_attn_patterns"] = enc_attn_patterns
    if isinstance(model_cfg["cross_self_pattern"], str):
        dec_self_attn_patterns = [model_cfg.pop("cross_self_pattern")] * num_blocks
    else:
        dec_self_attn_patterns = OmegaConf.to_container(model_cfg.pop("cross_self_pattern"))
    model_cfg["dec_self_attn_patterns"] = dec_self_attn_patterns
    if isinstance(model_cfg["cross_pattern"], str):
        dec_cross_attn_patterns = [model_cfg.pop("cross_pattern")] * num_blocks
    else:
        dec_cross_attn_patterns = OmegaConf.to_container(model_cfg.pop("cross_pattern"))
    model_cfg["dec_cross_attn_patterns"] = dec_cross_attn_patterns

    model = CuboidTransformerModelEarthNet2021x(**model_cfg)
    return model

pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "earthnet_w_meso", "earthformer_earthnet_v1.yaml")
pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
cfg.model.weather_conditioning_loc="early"
cfg.model.weather_conditioning="cat"
cfg.model.weather_conditioning_channels=24
model = config_cuboid_transformer(cfg=cfg)
batch_size = 2
context = torch.rand((batch_size, 10, 128, 128, 4))
target = torch.rand((batch_size, 20, 128, 128, 4))
cond = torch.rand((batch_size, 30, cfg.model.weather_conditioning_channels))
pred = model(context, cond, verbose=True)
torch.mean(pred).backward()
