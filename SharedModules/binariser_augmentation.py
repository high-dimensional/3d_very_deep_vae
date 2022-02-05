from collections.abc import Iterable
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.intensity.array import (
    AdjustContrast,
    GaussianSharpen,
    GaussianSmooth,
    MaskIntensity,
    NormalizeIntensity,
    ScaleIntensity,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    ShiftIntensity,
    ThresholdIntensity,
)
from monai.utils import dtype_torch_to_numpy, ensure_tuple_size
from monai.transforms.spatial.array import Resize


class Binariser(MapTransform):
    """

    """

    def __init__(self, keys: KeysCollection, min_val=0.5) -> None:
        """
        Args:
        """
        super().__init__(keys)

        self.min_val = min_val

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key][d[key] > self.min_val] = 1
        return d