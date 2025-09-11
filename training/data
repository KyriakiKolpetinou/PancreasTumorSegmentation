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
