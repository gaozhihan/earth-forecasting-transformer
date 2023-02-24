import numpy as np
import torch
from einops import rearrange
from earthformer.datasets.earthnet.earthnet21x_dataloader import EarthNet2021XDataset
from earthformer.datasets.earthnet.visualization import vis_earthnet_seq


dataset = EarthNet2021XDataset(EarthNet2021XDataset.default_train_dir)
idx = 0
layout = "NTHWC"
data = dataset[idx]["dynamic"][0]
data = rearrange(data, f"{' '.join('NTCHW')} -> {' '.join(layout)}")

vis_earthnet_seq(
    context_np=data,
    target_np=data,
    pred_np=data,
    ncols=10,
    layout=layout,
    variable="rgb",
    vegetation_mask=None,
    cloud_mask=True,
    save_path="tmp_earthnet2021x.png",)
