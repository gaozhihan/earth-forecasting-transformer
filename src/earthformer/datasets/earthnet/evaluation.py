"""
Code is adapted from
https://github.com/earthnet2021/earthnet-models-pytorch/blob/b87d71f4b082f68fe7b0ead303b87b75391d369b/earthnet_models_pytorch/setting/nnse_metric.py
and

"""
import numpy as np
from scipy.stats import hmean
import torch
from torchmetrics import Metric
from einops import rearrange
from .earthnet_toolkit.parallel_score import CubeCalculator as EN_CubeCalculator
from ...metrics.torchmetrics_wo_compute import MetricsUpdateWithoutCompute


class NNSEVeg(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False, process_group=None,
                 dist_sync_fn=None, lc_min=10., lc_max=40., ndvi_pred_idx=0, ndvi_targ_idx=0):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("nnse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(1e-6), dist_reduce_fx="sum")

        self.lc_min = lc_min
        self.lc_max = lc_max
        self.ndvi_pred_idx = ndvi_pred_idx
        self.ndvi_targ_idx = ndvi_targ_idx

    @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)  # accumulate the metrics
        self._forward_cache = None

        if self.compute_on_step:
            kwargs["just_return"] = True
            out_cache = self.update(*args, **kwargs)  # compute and return the rmse
            kwargs.pop("just_return", None)
            return out_cache

    def update(self, preds, batch, just_return=False):
        '''Any code needed to update the state given any inputs to the metric.

            args:
            preds (torch.Tensor): Prediction tensor of correct length with NDVI at channel index self.ndvi_pred_idx
            batch: (dict): dictionary from dataloader. Expects NDVI target tensor to be under key "dynamic", first entry, channel index self.ndvi_targ_idx.
        '''

        t_pred = preds.shape[1]

        lc = batch["landcover"]

        s2_mask = (batch["dynamic_mask"][0][:, -t_pred:, ...] < 1.).bool().type_as(preds)  # b t c h w
        lc_mask = ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool()).type_as(preds)  # b c h w

        ndvi_targ = batch["dynamic"][0][:, -t_pred:, self.ndvi_targ_idx, ...].unsqueeze(2)  # b t c h w

        ndvi_pred = preds[:, :, self.ndvi_pred_idx, ...].unsqueeze(2)  # b t c h w

        sum_squared_error = (((ndvi_targ - ndvi_pred) * s2_mask) ** 2).sum(1)  # b c h w
        mean_ndvi_targ = (ndvi_targ * s2_mask).sum(1).unsqueeze(1) / (s2_mask.sum(1).unsqueeze(1) + 1e-8)  # b t c h w

        sum_squared_deviation = (((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2).sum(1)  # b c h w

        nse = (1 - sum_squared_error / (sum_squared_deviation + 1e-8))  # b c h w

        nnse = (1 / (2 - nse)) * lc_mask  # b c h w

        n_obs = lc_mask.sum((1, 2, 3))  # b

        if just_return:
            cubenames = batch["cubename"]
            veg_score = 2 - 1 / (nnse.sum((1, 2, 3)) / n_obs)  # b
            return [{"name": cubenames[i], "veg_score": veg_score[i]} for i in range(len(cubenames))]
        else:
            self.nnse_sum += nnse.sum()
            self.n_obs += n_obs.sum()

    def compute(self):
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"veg_score": 2 - 1 / (self.nnse_sum / self.n_obs)}


class EarthNet2021Score(Metric):

    default_layout = "NHWCT"
    default_channel_axis = 3
    channels = 4

    def __init__(self,
                 layout: str = "NTHWC",
                 eps: float = 1e-4,
                 dist_sync_on_step: bool = False, ):
        super(EarthNet2021Score, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.layout = layout
        self.eps = eps

        self.add_state("MAD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("OLS",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("EMD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("SSIM",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        # does not count if NaN
        self.add_state("num_MAD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_OLS",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_EMD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_SSIM",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        r"""

        Parameters
        ----------
        pred, target:   torch.Tensor
            With the first dim as batch dim, and 4 channels (RGB and infrared)
        mask:   torch.Tensor
            With the first dim as batch dim, and 1 channel
        """
        pred_np = rearrange(pred.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        target_np = rearrange(target.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        # layout = "NHWCT"
        if mask is None:
            mask_np = np.ones_like(target_np)
        else:
            mask_np = torch.repeat_interleave(
                rearrange(1 - mask.detach(), f"{self.einops_layout} -> {self.einops_default_layout}"),
                repeats=self.channels, dim=self.default_channel_axis).cpu().numpy()
        for preds, targs, masks in zip(pred_np, target_np, mask_np):
            # Code is adapted from `load_file()` in ./earthnet_toolkit/parallel_score.py
            preds[preds < 0] = 0
            preds[preds > 1] = 1

            targs[np.isnan(targs)] = 0
            targs[targs > 1] = 1
            targs[targs < 0] = 0

            ndvi_preds = ((preds[:, :, 3, :] - preds[:, :, 2, :]) / (preds[:, :, 3, :] + preds[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_targs = ((targs[:, :, 3, :] - targs[:, :, 2, :]) / (targs[:, :, 3, :] + targs[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_masks = masks[:, :, 0, :][:, :, np.newaxis, :]
            # Code is adapted from `get_scores()` in ./earthnet_toolkit/parallel_score.py
            debug_info = {}
            mad, debug_info["MAD"] = EN_CubeCalculator.MAD(preds, targs, masks)
            ols, debug_info["OLS"] = EN_CubeCalculator.OLS(ndvi_preds, ndvi_targs, ndvi_masks)
            emd, debug_info["EMD"] = EN_CubeCalculator.EMD(ndvi_preds, ndvi_targs, ndvi_masks)
            ssim, debug_info["SSIM"] = EN_CubeCalculator.SSIM(preds, targs, masks)
            # does not count if NaN
            if mad is not None and not np.isnan(mad):
                self.MAD += mad
                self.num_MAD += 1
            if ols is not None and not np.isnan(ols):
                self.OLS += ols
                self.num_OLS += 1
            if emd is not None and not np.isnan(emd):
                self.EMD += emd
                self.num_EMD += 1
            if ssim is not None and not np.isnan(ssim):
                self.SSIM += ssim
                self.num_SSIM += 1

    def compute(self):
        MAD_mean = (self.MAD / (self.num_MAD + self.eps)).cpu().item()
        OLS_mean = (self.OLS / (self.num_OLS + self.eps)).cpu().item()
        EMD_mean = (self.EMD / (self.num_EMD + self.eps)).cpu().item()
        SSIM_mean = (self.SSIM / (self.num_SSIM + self.eps)).cpu().item()
        ENS = hmean([MAD_mean, OLS_mean, EMD_mean, SSIM_mean])
        return {
            "MAD": MAD_mean,
            "OLS": OLS_mean,
            "EMD": EMD_mean,
            "SSIM":SSIM_mean,
            "EarthNetScore": ENS,
        }


class EarthNet2021ScoreUpdateWithoutCompute(MetricsUpdateWithoutCompute):

    default_layout = "NHWCT"
    default_channel_axis = 3
    channels = 4

    def __init__(self,
                 layout: str = "NTHWC",
                 eps: float = 1e-4,
                 dist_sync_on_step: bool = False, ):
        super(EarthNet2021ScoreUpdateWithoutCompute, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.layout = layout
        self.eps = eps

        self.add_state("MAD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("OLS",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("EMD",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("SSIM",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        # does not count if NaN
        self.add_state("num_MAD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_OLS",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_EMD",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("num_SSIM",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    @property
    def einops_default_layout(self):
        if not hasattr(self, "_einops_default_layout"):
            self._einops_default_layout = " ".join(self.default_layout)
        return self._einops_default_layout

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        r"""

        Parameters
        ----------
        pred, target:   torch.Tensor
            With the first dim as batch dim, and 4 channels (RGB and infrared)
        mask:   torch.Tensor
            With the first dim as batch dim, and 1 channel
        """
        pred_np = rearrange(pred.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        target_np = rearrange(target.detach(), f"{self.einops_layout} -> {self.einops_default_layout}").cpu().numpy()
        # layout = "NHWCT"
        if mask is None:
            mask_np = np.ones_like(target_np)
        else:
            mask_np = torch.repeat_interleave(
                rearrange(1 - mask.detach(), f"{self.einops_layout} -> {self.einops_default_layout}"),
                repeats=self.channels, dim=self.default_channel_axis).cpu().numpy()
        for preds, targs, masks in zip(pred_np, target_np, mask_np):
            # Code is adapted from `load_file()` in ./earthnet_toolkit/parallel_score.py
            preds[preds < 0] = 0
            preds[preds > 1] = 1

            targs[np.isnan(targs)] = 0
            targs[targs > 1] = 1
            targs[targs < 0] = 0

            ndvi_preds = ((preds[:, :, 3, :] - preds[:, :, 2, :]) / (preds[:, :, 3, :] + preds[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_targs = ((targs[:, :, 3, :] - targs[:, :, 2, :]) / (targs[:, :, 3, :] + targs[:, :, 2, :] + 1e-6))[:,
                         :, np.newaxis, :]
            ndvi_masks = masks[:, :, 0, :][:, :, np.newaxis, :]
            # Code is adapted from `get_scores()` in ./earthnet_toolkit/parallel_score.py
            debug_info = {}
            mad, debug_info["MAD"] = EN_CubeCalculator.MAD(preds, targs, masks)
            ols, debug_info["OLS"] = EN_CubeCalculator.OLS(ndvi_preds, ndvi_targs, ndvi_masks)
            emd, debug_info["EMD"] = EN_CubeCalculator.EMD(ndvi_preds, ndvi_targs, ndvi_masks)
            ssim, debug_info["SSIM"] = EN_CubeCalculator.SSIM(preds, targs, masks)
            # does not count if NaN
            if mad is not None and not np.isnan(mad):
                self.MAD += mad
                self.num_MAD += 1
            if ols is not None and not np.isnan(ols):
                self.OLS += ols
                self.num_OLS += 1
            if emd is not None and not np.isnan(emd):
                self.EMD += emd
                self.num_EMD += 1
            if ssim is not None and not np.isnan(ssim):
                self.SSIM += ssim
                self.num_SSIM += 1

    def compute(self):
        MAD_mean = (self.MAD / (self.num_MAD + self.eps)).cpu().item()
        OLS_mean = (self.OLS / (self.num_OLS + self.eps)).cpu().item()
        EMD_mean = (self.EMD / (self.num_EMD + self.eps)).cpu().item()
        SSIM_mean = (self.SSIM / (self.num_SSIM + self.eps)).cpu().item()
        ENS = hmean([MAD_mean, OLS_mean, EMD_mean, SSIM_mean])
        return {
            "MAD": MAD_mean,
            "OLS": OLS_mean,
            "EMD": EMD_mean,
            "SSIM":SSIM_mean,
            "EarthNetScore": ENS,
        }