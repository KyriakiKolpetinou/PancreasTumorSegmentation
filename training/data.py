import numpy as np
import lightning as L
import torch
from monai.transforms import (
    AdjustContrast,
    CastToTyped,
    Compose,
    MapTransform,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandomizableTransform,
    RandRotated,
    RandScaleIntensityd,
    RandSimulateLowResolutiond,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)

#data loading

class LoadNpzd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        
        image_npz = np.load(data["image"])
        label_npz = np.load(data["label"])

        image = image_npz["arr_0"]  # shape: (1, D, H, W)
        label = label_npz["arr_0"]

        label[label < 0] = 0  # optional cleanup

        return {
            "image": image.astype(np.float32),
            "label": label.astype(np.uint8),
        }

#custom transforms 

class GammaAugmentationd(RandomizableTransform):
    def __init__(self, keys, gamma_range=(0.5, 2.0), prob=0.1, invert_prob=0.15):
        self.gamma_range = gamma_range
        self.prob = prob
        self.keys = keys
        self.invert_prob = invert_prob

    def randomize(self):
        self.gamma = self.R.uniform(low=self.gamma_range[0], high=self.gamma_range[1])
        self.invert = self.R.rand() < self.invert_prob

    def __call__(self, data):
        self.randomize()
        if self.R.rand() < self.prob:
            adjustContrast = AdjustContrast(gamma=self.gamma, invert_image=self.invert)
            for key in self.keys:
                data[key] = adjustContrast(data[key])
        return data


class RandContrastd(RandomizableTransform):
    def __init__(self, keys, prob=0.1, factor_range=(0.65, 1.5)):
        self.keys = keys
        self.prob = prob
        self.factor_range = factor_range

    def randomize(self):
        self.factor = self.R.uniform(
            low=self.factor_range[0], high=self.factor_range[1]
        )

    def __call__(self, data):
        self.randomize()
        if self.R.rand() < self.prob:
            for key in self.keys:
                intensity_min, intensity_max = data[key].min(), data[key].max()
                data[key] = data[key] * self.factor
                data[key] = np.clip(data[key], intensity_min, intensity_max)
        return data


#data loaders

class PanoramaDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 2,
        train_size: float = 0.7,
        val_size: float = 0.3,
        train_patch_size: tuple[int, int, int] = (40, 224, 224),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_patch_size = train_patch_size
        self.train_size = train_size
        self.val_size = val_size

    def get_transforms(self, mode):
        keys = ["image", "label"] if mode != "test" else ["image"]
        base_keys = keys.copy()
        keys.append("filename")

        load_transforms = [
            LoadNpzd(keys=base_keys),
            ToTensord(keys=base_keys),
            SpatialPadd(keys=base_keys, spatial_size=self.train_patch_size),
        ]


        augmentations = [
            RandCropByLabelClassesd(
        keys=["image","label"],
        label_key="label",
        ratios=[0.2, 0.3, 0.5],   # [bg, pancreas, tumor]
        spatial_size=self.train_patch_size,
        num_samples=1,
               image_key="image",
               num_classes=3,
               image_threshold=0,  # no thresholding
               ),
            RandRotated(
                keys=["image", "label"],
                range_x=(0.5),
                range_y=(0.5),
                range_z=(0.5),
                prob=0.2,
                keep_size=True,
                mode=("trilinear", "nearest"),
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.7,
                max_zoom=1.4,
                prob=0.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.1,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandContrastd(keys=["image"], prob=0.15),
            RandSimulateLowResolutiond(
                keys=["image"],
                prob=0.125,
                downsample_mode="nearest",
                upsample_mode="trilinear",
                zoom_range=(0.5, 1),
            ),
            GammaAugmentationd(
                keys=["image"],
                gamma_range=(0.7, 1.5),
                prob=0.15,
                invert_prob=0.15,
            ),
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
        ]

        other_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8))
        ] if mode != "test" else [CastToTyped(keys=["image"], dtype=(torch.float32))]

        if mode == "train":
            return Compose(load_transforms + augmentations + other_transforms)
        else:
            return Compose(load_transforms + other_transforms)

    def setup(self, stage=None):
        image_filepaths = sorted(glob.glob(os.path.join(self.data_dir, "imagesTr", "*")))
        label_filepaths = sorted(glob.glob(os.path.join(self.data_dir, "labelsTr", "*")))

        data_pairs = list(zip(image_filepaths, label_filepaths))
        random.shuffle(data_pairs)
        image_filepaths, label_filepaths = zip(*data_pairs)

        num_samples = len(image_filepaths)
        train_end = int(num_samples * self.train_size)

        train_files = [
    {"image": image, "label": label, "filename": os.path.basename(image)}
    for image, label in zip(image_filepaths[:train_end], label_filepaths[:train_end])
]

        val_files = [
    {"image": image, "label": label, "filename": os.path.basename(image)}
    for image, label in zip(image_filepaths[train_end:], label_filepaths[train_end:])
]


        self.train_ds = Dataset(data=train_files, transform=self.get_transforms("train"))
        self.val_ds = Dataset(data=val_files, transform=self.get_transforms("val"))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True,
                          collate_fn=pad_list_data_collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=pad_list_data_collate, pin_memory=True)
