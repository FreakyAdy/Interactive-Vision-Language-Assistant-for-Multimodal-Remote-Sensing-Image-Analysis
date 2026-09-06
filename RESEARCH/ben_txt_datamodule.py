from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from safetensors.numpy import load as safetensor_load
from torch.utils.data import Dataset, DataLoader

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

_s1_bandnames = ["VV", "VH"]
_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
_predefined_bandcombinations = {
    "RGB": ["B04", "B03", "B02"],
    "S2-10m20m": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "S1S2-10m20m": ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "all":  _s1_bandnames + _s2_bandnames,
}

"""
This file contains band statistics for BigEarthNet v2 (including S1 stats from v1) after 
interpolating the images to 120x120 with nearest neighbor.
The statistics are calculated based on the images of the official train split.
"""

means = {
    "B01": 361.0767822265625,
    "B02": 438.3720703125,
    "B03": 614.0556640625,
    "B04": 588.4096069335938,
    "B05": 942.8433227539062,
    "B06": 1769.931640625,
    "B07": 2049.551513671875,
    "B08": 2193.2919921875,
    "B09": 2241.455322265625,
    "B11": 1568.226806640625,
    "B12": 997.7324829101562,
    "B8A": 2235.556640625,
    "VH": -19.352558135986328,
    "VV": -12.643863677978516,
}
stds = {
    "B01": 575.0687255859375,
    "B02": 607.02685546875,
    "B03": 603.2968139648438,
    "B04": 684.56884765625,
    "B05": 738.4326782226562,
    "B06": 1100.4560546875,
    "B07": 1275.805419921875,
    "B08": 1369.3717041015625,
    "B09": 1316.393310546875,
    "B11": 1070.1612548828125,
    "B12": 813.5276489257812,
    "B8A": 1356.5440673828125,
    "VH": 5.590505599975586,
    "VV": 5.133493900299072,
}


def default_train_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean, std),
        ]
    )


def default_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean, std),
        ]
    )

def collate_mixed(batch):
    images = []
    texts = []
    outputs = []

    for item in batch:
        images.append(item["image_input"])
        texts.append(item["text_input"])
        outputs.append(item["reference_output"])

    # stack the images into a tensor batch
    images = torch.stack(images, dim=0)

    return {
        "image_input": images,
        "text_input": texts,  # list of strings
        "reference_output": outputs,  # list of arbitrary items
    }


class BENImageReader:
    def __init__(
            self,
            image_lmdb_file: Union[str, Path],
            metadata_file: Union[str, Path],
            bands: Iterable[str],
            img_size: int = 120,
            upsample_mode: str = "nearest",
            info_fn: Optional[Callable] = lambda x: x,
    ):
        self.img_size = img_size
        self.upsample_mode = upsample_mode
        self.image_lmdb_file = image_lmdb_file
        self.bands = bands
        self.env = None

        info_fn(f"Using bandorder {self.bands}")
        self.uses_s1 = any([x in _s1_bandnames for x in self.bands])
        self.uses_s2 = any([x in _s2_bandnames for x in self.bands])

        metadata = pd.read_parquet(metadata_file)
        self.mapping = {p: s for p, s in zip(metadata["patch_id"], metadata["s1_name"])}
        info_fn("S1-S2 mapping created")

    def stack_and_interpolate(
            self,
            data: Dict[str, np.ndarray],
    ) -> np.array:
        def _interpolate(img_data):
            if not img_data.shape[-2:] == (self.img_size, self.img_size):
                return F.interpolate(
                    torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                    (self.img_size, self.img_size),
                    mode=self.upsample_mode,
                    align_corners=True if self.upsample_mode in ["bilinear", "bicubic"] else None,
                ).squeeze()
            else:
                return torch.Tensor(np.float32(img_data))

        return torch.stack([_interpolate(data[x]) for x in self.bands])

    def open_env(self):
        if self.env is None:
            print("Opening LMDB environment ...")
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024 ** 3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )

    def __getitem__(self, key: str):
        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        if self.uses_s2:
            assert self.env is not None, "Environment not opened yet"
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            assert self.mapping is not None, "S1 bands are used, but no mapping is provided"
            s1_key = self.mapping[key]
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.stack_and_interpolate(img_data_dict)
        return img_data


class BENTxTDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet.txt.
    
    This dataset class loads the textual annotations from BigEarthNet.txt toegther with the satellite imagery (Sentinel-1 and Sentinel-2) from the BigEarthNet-v2.0 dataset (converted to LMDB format).
    It supports various filtering options to create custom dataset splits based on textual annotation metadata, such as type or category, or image metadata like country, season, and climate zone.
    """
    _expected_columns = {'s1_name', 'output', 'longitude', 'country', 'climate_zone', 'type', 'input', 'split', 'latitude', 'ID', 'patch_id', 'category', 'season'}

    def __init__(
            self,
            lmdb_file: Union[str, Path],
            metadata_file: Union[str, Path],
            bands: Iterable[str],
            img_size: int = 120,
            upsample_mode: str = "nearest",
            types: Optional[Iterable[str]] = None,
            categories: Optional[Iterable[str]] = None,
            countries: Optional[Iterable[str]] = None,
            seasons: Optional[Iterable[str]] = None,
            climate_zones: Optional[Iterable[str]] = None,
            transform: Optional[Callable] = None,
            splits: Optional[Iterable[str]] = None,
            point_token: Optional[str] = None,
            ref_token: Optional[str] = None,
            info_fn: Callable = lambda x: x,
    ):
        """
        Initialize the BigEarthNet.txt Dataset.
        
        Args:
            lmdb_file: Path to the LMDB file containing the BigEarthNet-v2.0 image data.
            metadata_file: Path to the BigEarthNet.txt Parquet file.
            bands: Band names to load. Can be a predefined combination key ('RGB', 'S2-10m20m', 'S1S2-10m20m', 'all') or an iterable of band names, i.e., ('B04', 'B03', 'B02').
            img_size: Target image size for interpolation (default: 120).
            upsample_mode: Interpolation mode for resizing ('nearest', 'bilinear', 'bicubic', etc.).
                Default: 'nearest'.
            types: Optional filter for annotation types (e.g., 'binary', 'mcq', 'captioning', 'bounding box').
            categories: Optional filter for annotation categories. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/8okbuKf) for possible type-category combinations or retrieve them by yourself using some kind of database tool on the Parquet file.
            countries: Optional filter for acquisition countries (e.g., 'Austria', 'Belgium', 'Finland', 'Ireland', 'Kosovo', 'Lithuania', 'Luxembourg', 'Portugal', 'Serbia', 'Switzerland').
            seasons: Optional filter for seasons (e.g., 'Spring', 'Summer', 'Fall', 'Winter').
            climate_zones: Optional filter for climate zones. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/3xLT8_u) for possible climate_zones values or retrieve them by yourself using some kind of database tool on the Parquet file. 
            transform: Optional torchvision transform to apply to images.
            splits: Optional filter for dataset splits ('train', 'validation', 'test', 'bench').
            point_token: Optional tuple of [start_token, end_token] to wrap <point> tags in text.
            ref_token: Optional tuple of [start_token, end_token] to wrap <ref> tags in text.
            info_fn: Optional callback function for logging information during initialization.
        """
        super().__init__()
        
        if isinstance(bands, str):
            assert bands in _predefined_bandcombinations, f"{bands} not in predefined options: {_predefined_bandcombinations.keys()}"
            bands = _predefined_bandcombinations[bands]
        elif isinstance(bands, Iterable):
            bands = list(bands)
        elif bands is None:
            bands = _predefined_bandcombinations['all']
        else:
            raise NotImplementedError(f"{bands} is not supported")

        self.image_reader = BENImageReader(lmdb_file, metadata_file, bands, img_size, upsample_mode, info_fn=info_fn)

        self.text_data = pd.read_parquet(metadata_file)

        # check the format of the text file
        assert self._expected_columns.issubset(set(self.text_data.columns)), f"The text data at {metadata_file} does not contain the expected columns"
        info_fn(f"Loaded text data with {len(self.text_data)} entries")
        if types is not None:
            self.text_data = self.text_data[self.text_data["type"].isin(types)]
        if categories is not None:
            self.text_data = self.text_data[self.text_data["category"].isin(categories)]
        if countries is not None:
            self.text_data = self.text_data[self.text_data["country"].isin(countries)]
        if seasons is not None:
            self.text_data = self.text_data[self.text_data["season"].isin(seasons)]
        if climate_zones is not None:
            self.text_data = self.text_data[self.text_data["climate_zone"].isin(climate_zones)]
        self.text_data = self.text_data.reset_index(drop=True)
        info_fn(f"After filtering, text data contains {len(self.text_data)} entries")

        if splits is not None:
            self.text_data = self.text_data[self.text_data["split"].isin(splits)].reset_index(drop=True)
            info_fn(f"Split {splits} text data contains {len(self.text_data)} entries")

        self.transform = transform
        self.point_token = ["", ""] if point_token is None else point_token
        assert len(self.point_token) == 2, "Point tokens must have length 2."
        self.ref_token = ["", ""] if ref_token is None else ref_token
        assert len(self.ref_token) == 2, "Reference tokens must have length 2."

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.text_data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing:
                - 'image_input': Tensor of shape (num_bands, img_size, img_size) containing the stacked bands, applying image transformations if transform is provided.
                - 'text_input': String containing the text query/caption with tokens replaced based on point_token and ref_token settings.
                - 'reference_output': The expected output/answer for the given input.
        """
        sample = self.text_data.iloc[idx]
        img_id = sample.patch_id
        img_data = self.image_reader[img_id]
        if self.transform is not None:
            img_data = self.transform(img_data)

        text_in = sample.input.replace("<ref>", self.ref_token[0]).replace("</ref>", self.ref_token[1])
        text_in = text_in.replace("<point>", self.point_token[0]).replace("</point>", self.point_token[1])

        if sample.type in {'binary', 'mcq', 'captioning', 'bounding box'}:
            output = sample.output
        else:
            raise NotImplementedError(f"{sample.type} is not supported")

        return {
            "image_input": img_data,
            "text_input": text_in,
            "reference_output": output,
        }


class BENTxTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for BigEarthNet.txt.
    
    This DataModule provides a structured interface for loading and preprocessing BigEarthNet.txt 
    data for use with PyTorch Lightning training loops. It automatically handles train, 
    validation, test, and benchmark dataset splits, with proper train/eval transforms and 
    DataLoader configuration.
    
    The module manages:
    - Automatic dataset setup for different training stages
    - Image preprocessing and normalization based on selected bands
    - DataLoader creation with appropriate batch sizes and worker processes
    - GPU pinning when CUDA is available
    
    Attributes:
        train_ds (BENTxTDataset): Training dataset instance.
        val_ds (BENTxTDataset): Validation dataset instance.
        test_ds (BENTxTDataset): Test dataset instance.
        bench_ds (BENTxTDataset): Benchmark dataset instance.
    """
    train_ds = None
    val_ds = None
    test_ds = None
    bench_ds = None

    def __init__(
            self,
            image_lmdb_file: Union[str, Path],
            metadata_file: Union[str, Path],
            bands: Optional[Union[Iterable[str], str]] = None,
            img_size: int = 120,
            upsample_mode: str = "nearest",
            types: Optional[Iterable[str]] = None,
            categories: Optional[Iterable[str]] = None,
            countries: Optional[Iterable[str]] = None,
            seasons: Optional[Iterable[str]] = None,
            climate_zones: Optional[Iterable[str]] = None,
            num_workers_dataloader: Optional[int] = 4,
            batch_size: Optional[int] = 16,
            image_transforms_train: Optional[Callable] = None,
            image_transforms_eval: Optional[Callable] = None,
            point_token: Iterable[str] = None,
            ref_token: Iterable[str] = None,
            info_fn: Optional[Callable] = lambda x: x,
    ):
        """
        Initialize the BigEarthNet.txt DataModule.
        
        Args:
            lmdb_file: Path to the LMDB file containing the BigEarthNet-v2.0 image data.
            metadata_file: Path to the BigEarthNet.txt Parquet file.
            bands: Band names to load. Can be a predefined combination key ('RGB', 'S2-10m20m', 'S1S2-10m20m', 'all') or an iterable of band names, i.e., ('B04', 'B03', 'B02').
            img_size: Target image size for interpolation (default: 120).
            upsample_mode: Interpolation mode for resizing ('nearest', 'bilinear', 'bicubic', etc.).
                Default: 'nearest'.
            types: Optional filter for annotation types (e.g., 'binary', 'mcq', 'captioning', 'bounding box').
            categories: Optional filter for annotation categories. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/KzrmYgF) for possible type-category combinations or retrieve them by yourself using some kind of database tool on the Parquet file.
            countries: Optional filter for acquisition countries (e.g., 'Austria', 'Belgium', 'Finland', 'Ireland', 'Kosovo', 'Lithuania', 'Luxembourg', 'Portugal', 'Serbia', 'Switzerland').
            seasons: Optional filter for seasons (e.g., 'Spring', 'Summer', 'Fall', 'Winter').
            climate_zones: Optional filter for climate zones. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/SUU1DwA) for possible climate_zones values or retrieve them by yourself using some kind of database tool on the Parquet file. 
            transform: Optional torchvision transform to apply to images.
            num_workers_dataloader: Number of worker processes for DataLoaders (default: 4).
                Set to 0 to disable multiprocessing.
            batch_size: Batch size for DataLoaders (default: 16).
            image_transforms_train: Custom image transforms for training. If None, uses default
                augmentations (resize, flip, normalize).
            image_transforms_eval: Custom image transforms for evaluation/validation. If None,
                uses simple normalization.
            point_token: Optional tuple of [start_token, end_token] to wrap <point> tags in text.
            ref_token: Optional tuple of [start_token, end_token] to wrap <ref> tags in text.
            info_fn: Optional callback function for logging during initialization.
        """
        super().__init__()
        self.num_workers_dataloader = num_workers_dataloader
        self.batch_size = batch_size
        self.pin_memory = torch.cuda.is_available()

        if isinstance(bands, str):
            assert bands in _predefined_bandcombinations, f"{bands} not in predefined options: {_predefined_bandcombinations.keys()}"
            self.bands = _predefined_bandcombinations[bands]
        elif isinstance(bands, Iterable):
            self.bands = list(bands)
        elif bands is None:
            self.bands = _predefined_bandcombinations['all']
        else:
            raise NotImplementedError(f"{bands} is not supported")

        self.ds_kwargs = {
            "lmdb_file": image_lmdb_file,
            "metadata_file": metadata_file,
            "bands": self.bands,
            "img_size": img_size,
            "upsample_mode": upsample_mode,
            "types": types,
            "categories": categories,
            "countries": countries,
            "seasons": seasons,
            "climate_zones": climate_zones,
            "point_token": point_token,
            "ref_token": ref_token,
            "info_fn": info_fn,
        }

        # set mean and std based on bands selected
        self.mean = [means[b] for b in self.bands]
        self.std = [stds[b] for b in self.bands]

        self.train_transforms = image_transforms_train if image_transforms_train is not None else default_train_transform(img_size, mean=self.mean, std=self.std)
        self.eval_transforms = image_transforms_eval if image_transforms_eval is not None else default_transform(img_size, mean=self.mean, std=self.std)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create train/val/test/bench datasets based on the specified stage.
        
        This method is called by PyTorch Lightning during trainer initialization.
        
        Args:
            stage: The training stage - one of 'fit', 'test', 'bench', or None. If None,
                all datasets are created. Default: None.
                - 'fit': Creates train and validation datasets
                - 'test': Creates test dataset (includes both 'test' and 'bench' splits)
                - 'bench': Creates benchmark dataset
        """
        if stage == "fit" or stage is None:
            self.train_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['train'],
                transform=self.train_transforms
            )
            self.val_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['validation'],
                transform=self.eval_transforms
            )
        if stage == "test" or stage is None:
            self.test_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['test', 'bench'],
                transform=self.eval_transforms
            )
        if stage == "bench" or stage is None:
            self.bench_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['bench'],
                transform=self.eval_transforms
            )



    def train_dataloader(self):
        """Create and return the training DataLoader with shuffling and augmentations."""
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers_dataloader, shuffle=True, pin_memory=self.pin_memory, collate_fn=collate_mixed)

    def val_dataloader(self):
        """Create and return the validation DataLoader without shuffling."""
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers_dataloader, shuffle=False, pin_memory=self.pin_memory, collate_fn=collate_mixed)

    def test_dataloader(self):
        """Create and return the test DataLoader (includes both 'test' and 'bench' splits)."""
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers_dataloader, shuffle=False, pin_memory=self.pin_memory, collate_fn=collate_mixed)

    def bench_dataloader(self):
        """Create and return the benchmark DataLoader."""
        return DataLoader(self.bench_ds, batch_size=self.batch_size, num_workers=self.num_workers_dataloader, shuffle=False, pin_memory=self.pin_memory, collate_fn=collate_mixed)
