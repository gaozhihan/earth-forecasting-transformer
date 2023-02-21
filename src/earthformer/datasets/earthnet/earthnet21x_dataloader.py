"""Code is adapted from https://github.com/earthnet2021/earthnet-models-pytorch/blob/b87d71f4b082f68fe7b0ead303b87b75391d369b/earthnet_models_pytorch/setting/en21x_data.py"""
from typing import Union, Optional, Sequence, Dict
import os
import warnings
import argparse
import copy
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import xarray as xr
from ...config import cfg


default_data_dir = os.path.join(cfg.datasets_dir, "earthnet2021x")

class EarthNet2021XDataset(Dataset):

    default_train_dir = os.path.join(default_data_dir, "train")
    default_iid_test_data_dir = os.path.join(default_data_dir, "iid")
    default_ood_test_data_dir = os.path.join(default_data_dir, "ood")
    default_extreme_test_data_dir = os.path.join(default_data_dir, "extreme")
    default_seasonal_test_data_dir = os.path.join(default_data_dir, "seasonal")

    def __init__(self, folder: Union[Path, str], fp16=False, s2_bands=["ndvi", "B02", "B03", "B04", "B8A"],
                 eobs_vars=['fg', 'hu', 'pp', 'qq', 'rr', 'tg', 'tn', 'tx'], eobs_agg=['mean', 'min', 'max'],
                 static_vars=['nasa_dem', 'alos_dem', 'cop_dem', 'esawc_lc', 'geom_cls'], start_month_extreme=None,
                 dl_cloudmask=False):
        if not isinstance(folder, Path):
            folder = Path(folder)

        self.filepaths = sorted(list(folder.glob("**/*.nc")))

        self.type = np.float16 if fp16 else np.float32

        self.s2_bands = s2_bands
        self.eobs_vars = eobs_vars
        self.eobs_agg = eobs_agg
        self.static_vars = static_vars
        self.start_month_extreme = start_month_extreme
        self.dl_cloudmask = dl_cloudmask

        self.eobs_mean = xr.DataArray(
            data=[8.90661030749754, 2.732927619847993, 77.54440854529798, 1014.330962704611, 126.47924227500346,
                  1.7713217310829938, 4.770701430461286, 13.567999825718509], coords={
                'variable': ['eobs_tg', 'eobs_fg', 'eobs_hu', 'eobs_pp', 'eobs_qq', 'eobs_rr', 'eobs_tn', 'eobs_tx']})
        self.eobs_std = xr.DataArray(
            data=[9.75620252236597, 1.4870108944469236, 13.511387994026359, 10.262645403460999, 97.05522895011327,
                  4.147967261223076, 9.044987677752898, 11.08198777356161], coords={
                'variable': ['eobs_tg', 'eobs_fg', 'eobs_hu', 'eobs_pp', 'eobs_qq', 'eobs_rr', 'eobs_tn', 'eobs_tx']})

        self.static_mean = xr.DataArray(data=[0.0, 0.0, 0.0, 0.0, 0.0], coords={
            'variable': ['nasa_dem', 'alos_dem', 'cop_dem', 'esawc_lc', 'geom_cls']})
        self.static_std = xr.DataArray(data=[500.0, 500.0, 500.0, 1.0, 1.0],
                                       coords={'variable': ['nasa_dem', 'alos_dem', 'cop_dem', 'esawc_lc', 'geom_cls']})

    def __getitem__(self, idx: int) -> dict:

        filepath = self.filepaths[idx]

        minicube = xr.open_dataset(filepath)

        if self.start_month_extreme:
            start_idx = {"march": 10, "april": 15, "may": 20, "june": 25, "july": 30}[self.start_month_extreme]
            minicube = minicube.isel(time=slice(5 * start_idx, 5 * (start_idx + 30)))

        # if minicube.s2_B02.shape[1] != 128:
        #     # print("lat", filepath, minicube.s2_B02.shape, minicube["s2_B02"].dropna(dim = "lat", how = "all").shape)
        #     new_lat = xr.DataArray(coords = {"lat": np.linspace(minicube.lat.values[0], minicube.lat.values[-1] if len(minicube.lat) > 2 else minicube.lat.values[0] - 0.023551999999654072, 128)}, dims = ("lat", ))
        #     new_mc = [minicube[[k for k in minicube.data_vars if k not in ["s2_B02", "s2_B03", "s2_B04", "s2_B8A", "alos_dem", "cop_dem", "srtm_dem", "esawc_lc", "geom_cls", "s2_SCL", "s2_mask"]]]]

        #     for var in ["s2_B02", "s2_B03", "s2_B04", "s2_B8A", "alos_dem", "cop_dem", "srtm_dem", "esawc_lc", "geom_cls", "s2_SCL", "s2_mask"]:
        #         if var in minicube:
        #             curr_mc = minicube[var].dropna(dim = "lat", how = "all")
        #             if len(curr_mc.lat) != 128:
        #                 curr_mc = curr_mc.interp_like(new_lat)
        #             else:
        #                 curr_mc["lat"] = new_lat
        #             #curr_mc["lat"] = minicube["s2_B02"].dropna(dim = "lat", how = "all").lat
        #             new_mc.append(curr_mc)
        #     for i in range(len(new_mc[1:])):
        #         if "lat" in new_mc[i+1].dims:
        #             new_mc[i+1]["lat"] = new_lat

        #     minicube = xr.merge(new_mc)

        # if minicube.s2_B02.shape[2] != 128:
        #     # print("lon", filepath, minicube.s2_B02.shape, minicube["s2_B02"].dropna(dim = "lon", how = "all").shape)
        #     new_lon = xr.DataArray(coords = {"lon": np.linspace(minicube.lon.values[0], minicube.lon.values[-1] if len(minicube.lon) > 2 else minicube.lon.values[0] + 0.037961000000000134, 128)}, dims = ("lon", ))
        #     new_mc = [minicube[[k for k in minicube.data_vars if k not in ["s2_B02", "s2_B03", "s2_B04", "s2_B8A", "alos_dem", "cop_dem", "nasa_dem", "esawc_lc", "geom_cls", "s2_SCL", "s2_mask"]]]]

        #     for var in ["s2_B02", "s2_B03", "s2_B04", "s2_B8A", "alos_dem", "cop_dem", "nasa_dem", "esawc_lc", "geom_cls", "s2_SCL", "s2_mask"]:
        #         if var in minicube:
        #             curr_mc = minicube[var].dropna(dim = "lon", how = "all")
        #             if len(curr_mc.lon) != 128:
        #                 curr_mc = curr_mc.interp_like(new_lon)
        #             else:
        #                 curr_mc["lon"] = new_lon
        #             #curr_mc["lon"] = minicube["s2_B02"].dropna(dim = "lon", how = "all").lon
        #             new_mc.append(curr_mc)
        #     for i in range(len(new_mc[1:])):
        #         if "lon" in new_mc[i+1].dims:
        #             new_mc[i+1]["lon"] = new_lon
        #     minicube = xr.merge(new_mc)

        nir = minicube.s2_B8A
        red = minicube.s2_B04

        ndvi = ((nir - red) / (nir + red + 1e-8))

        minicube["s2_ndvi"] = ndvi

        sen2arr = minicube[[f"s2_{b}" for b in self.s2_bands]].to_array("band").isel(time=slice(4, None, 5)).transpose(
            "time", "band", "lat", "lon").values

        sen2arr[np.isnan(sen2arr)] = 0.0  # Fill NaNs!!

        if self.dl_cloudmask:
            sen2mask = minicube.s2_dlmask.where(minicube.s2_dlmask > 0,
                                                4 * (~minicube.s2_SCL.isin([1, 2, 4, 5, 6, 7]))).isel(
                time=slice(4, None, 5)).transpose("time", "lat", "lon").values[:, None, ...]
            sen2mask[np.isnan(sen2mask)] = 4.
        else:
            sen2mask = minicube[["s2_mask"]].to_array("band").isel(time=slice(4, None, 5)).transpose("time", "band",
                                                                                                     "lat",
                                                                                                     "lon").values
            sen2mask[np.isnan(sen2mask)] = 4.

        eobs = ((minicube[[f'eobs_{v}' for v in self.eobs_vars]].to_array(
            "variable") - self.eobs_mean) / self.eobs_std).transpose("time", "variable")

        eobsarr = []
        if "mean" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").mean())
        if "min" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").min())
        if "max" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").max())
        if "std" in self.eobs_agg:
            eobsarr.append(eobs.coarsen(time=5, coord_func="max").std())

        eobsarr = np.concatenate(eobsarr, axis=1)

        eobsarr[np.isnan(eobsarr)] = 0.  # MAYBE BAD IDEA......

        # for static_var in self.static_vars + ["esawc_lc"]: # somehow sometimes a DEM might be missing..
        #     if static_var not in minicube:
        #         minicube[static_var] = xr.DataArray(data = np.full(shape = (len(minicube.lat), len(minicube.lon)), fill_value = np.NaN), coords = {"lat": minicube.lat, "lon": minicube.lon}, dims = ("lat", "lon"))

        staticarr = ((minicube[self.static_vars].to_array("variable") - self.static_mean) / self.static_std).transpose(
            "variable", "lat", "lon").values

        staticarr[np.isnan(staticarr)] = 0.  # MAYBE BAD IDEA......

        lc = minicube[['esawc_lc']].to_array("variable").transpose("variable", "lat", "lon").values  # c h w

        lc[np.isnan(lc)] = 0

        data = {
            "dynamic": [
                torch.from_numpy(sen2arr.astype(self.type)),
                torch.from_numpy(eobsarr.astype(self.type))
            ],
            "dynamic_mask": [
                torch.from_numpy(sen2mask.astype(self.type))
            ],
            "static": [
                torch.from_numpy(staticarr.astype(self.type))
            ],
            "static_mask": [],
            "landcover": torch.from_numpy(lc.astype(self.type)),
            "filepath": str(filepath),
            "cubename": filepath.stem
        }
        return data

    def __len__(self) -> int:
        return len(self.filepaths)


class EarthNet2021xLightningDataModule(LightningDataModule):

    def __init__(self,
                 train_data_dir: Union[Path, str] = None,
                 test_subset_name: Union[str, Sequence[str]] = ("iid", "ood"),
                 test_data_dir: Union[Union[Path, str], Sequence[Union[Path, str]]] = None,
                 val_ratio: float = 0.1,
                 train_val_split_seed: int = None,
                 fp16: bool = False,
                 dl_cloudmask: bool = False,
                 # datamodule_only
                 batch_size=1,
                 num_workers=multiprocessing.cpu_count(),):
        super(EarthNet2021xLightningDataModule, self).__init__()
        if train_data_dir is None:
            train_data_dir = EarthNet2021XDataset.default_train_dir
        self.train_data_dir = train_data_dir

        if test_subset_name is None:
            if not isinstance(test_data_dir, Sequence):
                self.test_data_dir_list = [test_data_dir, ]
            else:
                self.test_data_dir_list = list(test_data_dir)
            self.test_subset_name_list = [None, ] * len(self.test_data_dir_list)
        else:
            if isinstance(test_subset_name, str):
                self.test_subset_name_list = [test_subset_name, ]
            elif isinstance(test_subset_name, Sequence):
                self.test_subset_name_list = list(test_subset_name)
            else:
                raise ValueError(f"Invalid type of test_subset_name {type(test_subset_name)}")
            self.test_data_dir_list = []
            for test_subset_name in self.test_subset_name_list:
                if test_subset_name == "iid":
                    test_data_dir = EarthNet2021XDataset.default_iid_test_data_dir
                elif test_subset_name == "ood":
                    test_data_dir = EarthNet2021XDataset.default_ood_test_data_dir
                elif test_subset_name == "extreme":
                    test_data_dir = EarthNet2021XDataset.default_extreme_test_data_dir
                elif test_subset_name == "seasonal":
                    test_data_dir = EarthNet2021XDataset.default_seasonal_test_data_dir
                else:
                    raise ValueError(f"Invalid test_subset_name {test_subset_name}")
                self.test_data_dir_list.append(test_data_dir)

        self.val_ratio = val_ratio
        self.train_val_split_seed = train_val_split_seed

        self.fp16 = fp16
        self.dl_cloudmask = dl_cloudmask
        # datamodule_only
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        assert os.path.exists(self.train_data_dir), "EarthNet2021 training set not found!"
        for test_data_dir in self.test_data_dir_list:
            assert os.path.exists(test_data_dir), f"EarthNet2021 test set at {test_data_dir} not found!"

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_val_data = EarthNet2021XDataset(
                folder=self.train_data_dir,
                fp16=self.fp16,
                dl_cloudmask=self.dl_cloudmask,)
            val_size = int(self.val_ratio * len(train_val_data))
            train_size = len(train_val_data) - val_size

            if self.train_val_split_seed is not None:
                rnd_generator_dict = dict(generator=torch.Generator().manual_seed(self.train_val_split_seed))
            else:
                rnd_generator_dict = {}
            self.earthnet_train, self.earthnet_val = random_split(
                train_val_data, [train_size, val_size],
                **rnd_generator_dict)

        if stage in (None, "test"):
            self.earthnet_test_list = [
                EarthNet2021XDataset(
                    folder=test_data_dir,
                    fp16=self.fp16,
                    dl_cloudmask=self.dl_cloudmask,)
                for test_subset_name, test_data_dir in
                zip(self.test_subset_name_list, self.test_data_dir_list)]

        if stage in (None, "predict"):
            self.earthnet_predict_list = [
                EarthNet2021XDataset(
                    folder=test_data_dir,
                    fp16=self.fp16,
                    dl_cloudmask=self.dl_cloudmask,)
                for test_subset_name, test_data_dir in
                zip(self.test_subset_name_list, self.test_data_dir_list)]

    @property
    def num_train_samples(self):
        return len(self.earthnet_train)

    @property
    def num_val_samples(self):
        return len(self.earthnet_val)

    @property
    def num_test_samples(self):
        if len(self.earthnet_test_list) == 1:
            return len(self.earthnet_test_list[0])
        else:
            return [len(earthnet_test) for earthnet_test in self.earthnet_test_list]

    @property
    def num_predict_samples(self):
        if len(self.earthnet_predict_list) == 1:
            return len(self.earthnet_predict_list[0])
        else:
            return [len(earthnet_predict) for earthnet_predict in self.earthnet_predict_list]

    def train_dataloader(self):
        return DataLoader(self.earthnet_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.earthnet_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if len(self.earthnet_test_list) == 1:
            return DataLoader(self.earthnet_test_list[0], batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers)
        else:
            return [DataLoader(earthnet_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                    for earthnet_test in self.earthnet_test_list]

    def predict_dataloader(self):
        if len(self.earthnet_predict_list) == 1:
            return DataLoader(self.earthnet_predict_list[0], batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers)
        else:
            return [
                DataLoader(earthnet_predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                for earthnet_predict in self.earthnet_predict_list]


def get_EarthNet2021x_dataloaders(
        train_data_dir: Union[Path, str] = None,
        test_subset_name: Union[str, Sequence[str]] = ("iid", "ood"),
        test_data_dir: Union[Union[Path, str], Sequence[Union[Path, str]]] = None,
        val_ratio: float = 0.1,
        train_val_split_seed: int = None,
        fp16: bool = False,
        dl_cloudmask: bool = False,
        batch_size=1,
        num_workers=multiprocessing.cpu_count(),):
    if train_data_dir is None:
        train_data_dir = EarthNet2021XDataset.default_train_dir
    if test_subset_name is None:
        if not isinstance(test_data_dir, Sequence):
            test_data_dir_list = [test_data_dir, ]
        else:
            test_data_dir_list = list(test_data_dir)
        test_subset_name_list = [None, ] * len(test_data_dir_list)
    else:
        if isinstance(test_subset_name, str):
            test_subset_name_list = [test_subset_name, ]
        elif isinstance(test_subset_name, Sequence):
            test_subset_name_list = list(test_subset_name)
        else:
            raise ValueError(f"Invalid type of test_subset_name {type(test_subset_name)}")
        test_data_dir_list = []
        for test_subset_name in test_subset_name_list:
            if test_subset_name == "iid":
                test_data_dir = EarthNet2021XDataset.default_iid_test_data_dir
            elif test_subset_name == "ood":
                test_data_dir = EarthNet2021XDataset.default_ood_test_data_dir
            elif test_subset_name == "extreme":
                test_data_dir = EarthNet2021XDataset.default_extreme_test_data_dir
            elif test_subset_name == "seasonal":
                test_data_dir = EarthNet2021XDataset.default_seasonal_test_data_dir
            else:
                raise ValueError(f"Invalid test_subset_name {test_subset_name}")
            test_data_dir_list.append(test_data_dir)

    train_val_data = EarthNet2021XDataset(
        folder=train_data_dir,
        fp16=fp16,
        dl_cloudmask=dl_cloudmask,)
    val_size = int(val_ratio * len(train_val_data))
    train_size = len(train_val_data) - val_size

    if train_val_split_seed is not None:
        rnd_generator_dict = dict(generator=torch.Generator().manual_seed(train_val_split_seed))
    else:
        rnd_generator_dict = {}
    earthnet_train, earthnet_val = random_split(
        train_val_data, [train_size, val_size],
        **rnd_generator_dict)

    earthnet_test_list = [
        EarthNet2021XDataset(
            folder=test_data_dir,
            fp16=fp16,
            dl_cloudmask=dl_cloudmask,)
        for test_subset_name, test_data_dir in
        zip(test_subset_name_list, test_data_dir_list)]

    num_test_samples = [len(earthnet_test) for earthnet_test in earthnet_test_list]
    test_dataloader = [DataLoader(earthnet_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                       for earthnet_test in earthnet_test_list]

    return {
        "train_dataloader": DataLoader(earthnet_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val_dataloader": DataLoader(earthnet_val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_dataloader": test_dataloader,
        "num_train_samples": len(earthnet_train),
        "num_val_samples": len(earthnet_val),
        "num_test_samples": num_test_samples,
    }
