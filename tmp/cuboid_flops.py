import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from einops import rearrange
from omegaconf import OmegaConf, DictConfig, ListConfig
import os
from earthformer.cuboid_transformer.cuboid_transformer_unet_dec import CuboidTransformerAuxModel
from earthformer.utils.utils import count_num_params


test_dataset_name = "earthnet2021"

if test_dataset_name == "earthnet2021":
    # cfg_file_path = os.path.join(os.path.dirname(__file__), "..",
    #                              "scripts", "cuboid_transformer", "earthnet_w_meso", "earthformer_earthnet_v1.yaml")
    # cfg_file_path = os.path.join(os.path.dirname(__file__), "..",
    #                              "scripts", "cuboid_transformer", "earthnet_w_meso", "cfg.yaml")
    cfg_file_path = os.path.join(os.path.dirname(__file__), "cuboid_flops_cfg.yaml")
else:
    raise NotImplementedError

if test_dataset_name == "earthnet2021":
    def get_model_config():
        cfg = OmegaConf.create()
        cfg.data_channels = 4
        cfg.input_shape = (10, 128, 128, cfg.data_channels)
        cfg.target_shape = (20, 128, 128, cfg.data_channels)

        cfg.base_units = 64
        cfg.block_units = None  # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True

        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        # different from CuboidTransformerModel, no arg `dec_use_first_self_attn=False`
        cfg.auxiliary_channels = 7  # 5 from mesodynamic, 1 from highresstatic, 1 from mesostatic
        cfg.unet_dec_cross_mode = "up"
        return cfg
else:
    raise ValueError

model_cfg = get_model_config()
if cfg_file_path is not None:
    oc_from_file = OmegaConf.load(open(cfg_file_path, "r"))
    model_cfg = OmegaConf.merge(get_model_config(), oc_from_file.model)
model_cfg = OmegaConf.to_object(model_cfg)
model_cfg["checkpoint_level"] = 0

num_blocks = len(model_cfg["enc_depth"])
if isinstance(model_cfg["self_pattern"], str):
    enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
else:
    enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
if isinstance(model_cfg["cross_self_pattern"], str):
    dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
else:
    dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
if isinstance(model_cfg["cross_pattern"], str):
    dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
else:
    dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

model = CuboidTransformerAuxModel(
    input_shape=model_cfg["input_shape"],
    target_shape=model_cfg["target_shape"],
    base_units=model_cfg["base_units"],
    block_units=model_cfg["block_units"],
    scale_alpha=model_cfg["scale_alpha"],
    enc_depth=model_cfg["enc_depth"],
    dec_depth=model_cfg["dec_depth"],
    enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
    dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
    dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
    downsample=model_cfg["downsample"],
    downsample_type=model_cfg["downsample_type"],
    enc_attn_patterns=enc_attn_patterns,
    dec_self_attn_patterns=dec_self_attn_patterns,
    dec_cross_attn_patterns=dec_cross_attn_patterns,
    dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
    num_heads=model_cfg["num_heads"],
    attn_drop=model_cfg["attn_drop"],
    proj_drop=model_cfg["proj_drop"],
    ffn_drop=model_cfg["ffn_drop"],
    upsample_type=model_cfg["upsample_type"],
    ffn_activation=model_cfg["ffn_activation"],
    gated_ffn=model_cfg["gated_ffn"],
    norm_layer=model_cfg["norm_layer"],
    # global vectors
    num_global_vectors=model_cfg["num_global_vectors"],
    use_dec_self_global=model_cfg["use_dec_self_global"],
    dec_self_update_global=model_cfg["dec_self_update_global"],
    use_dec_cross_global=model_cfg["use_dec_cross_global"],
    use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
    use_global_self_attn=model_cfg["use_global_self_attn"],
    separate_global_qkv=model_cfg["separate_global_qkv"],
    global_dim_ratio=model_cfg["global_dim_ratio"],
    # initial_downsample
    initial_downsample_type=model_cfg["initial_downsample_type"],
    initial_downsample_activation=model_cfg["initial_downsample_activation"],
    # initial_downsample_type=="stack_conv"
    initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
    initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
    initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
    initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
    # misc
    padding_type=model_cfg["padding_type"],
    checkpoint_level=model_cfg["checkpoint_level"],
    pos_embed_type=model_cfg["pos_embed_type"],
    use_relative_pos=model_cfg["use_relative_pos"],
    self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
    # initialization
    attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
    ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
    conv_init_mode=model_cfg["conv_init_mode"],
    down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
    norm_init_mode=model_cfg["norm_init_mode"],
    # different from CuboidTransformerModel, no arg `dec_use_first_self_attn=False`
    auxiliary_channels=model_cfg["auxiliary_channels"],
    unet_dec_cross_mode=model_cfg["unet_dec_cross_mode"],
)
model.train()

if test_dataset_name == "earthnet2021":
    # # SEVIR
    in_len = 10
    out_len = 20
    height = 128
    width = 128
    channels = 4
    h_aux = height
    w_aux = width
    c_aux = 7
else:
    raise NotImplementedError

device = torch.device("cpu")
# device = torch.device("cuda:0")

in_seq = torch.rand((1, in_len, height, width, channels)).to(device)
out_seq = torch.rand((1, out_len, height, width, channels)).to(device)
in_aux = torch.rand((1, in_len, h_aux, w_aux, c_aux)).to(device)
out_aux = torch.rand((1, out_len, h_aux, w_aux, c_aux)).to(device)
flops = FlopCountAnalysis(model=model.to(device),
                          inputs=(in_seq, in_aux, out_aux))
output = model.to(device)(in_seq, in_aux, out_aux, verbose=True)
# print(f"flops = {flops.total()}")
print(f"flops = {flops.total():,}")
# print(f"#params = {count_num_params(model)}")
print(f"#params = {count_num_params(model):,}")
