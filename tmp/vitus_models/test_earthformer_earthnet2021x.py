import os
from omegaconf import OmegaConf
import torch
from earthformer.config import cfg
from earthformer.cuboid_transformer.cuboid_transformer_earthnet2021x import CuboidTransformerModelEarthNet2021x
from earthformer.datasets.earthnet.conditional_layer import (
    Identity_Conditioning, Cat_Conditioning,
    Cat_Project_Conditioning, CrossAttention_Conditioning, FiLMBlock)


def test_cond_layers():
    x_channels = 40
    c_channels = 120
    hidden_channels = 64
    # model = Cat_Project_Conditioning(x_channels=x_channels, c_channels=c_channels)
    # model = FiLMBlock(n_x=x_channels, n_c=c_channels, n_hid=hidden_channels)
    model = CrossAttention_Conditioning(
        x_channels=x_channels, c_channels=c_channels, n_tokens_c=1,
        hidden_dim=hidden_channels, n_heads=4, mlp_after_attn=False)
    x = torch.rand((2, 40, 128, 128))
    cond = torch.rand((2, 120, 128, 128))
    out = model(x, cond)
    print(f"out.shape = {out.shape}")

def config_cuboid_transformer(cfg):
    model_cfg = OmegaConf.to_object(cfg.model)
    model_cfg.pop("data_channels")
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

def test_earthformer_earthnet2021x():
    pretrained_cfg_path = os.path.join(cfg.root_dir, "scripts", "cuboid_transformer", "earthnet2021x", "cond_weather_data", "cfg.yaml")
    pretrained_cfg = OmegaConf.load(open(pretrained_cfg_path, "r"))
    pretrained_cfg.model.weather_conditioning_loc="early"
    pretrained_cfg.model.weather_conditioning="cat"
    pretrained_cfg.model.weather_conditioning_channels=24
    model = config_cuboid_transformer(cfg=pretrained_cfg)
    batch_size = 2
    context = torch.rand((batch_size, 10, 128, 128, 5))
    target = torch.rand((batch_size, 20, 128, 128, 5))
    cond = torch.rand((batch_size, 30, 2, 2, pretrained_cfg.model.weather_conditioning_channels))
    pred = model(context, cond, verbose=True)
    torch.mean(pred).backward()

test_earthformer_earthnet2021x()
