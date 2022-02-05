from collections.abc import Iterable
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
from monai.config import DtypeLike, KeysCollection, NdarrayTensor
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.utils import dtype_torch_to_numpy, ensure_tuple_size


class CropNIIByGivenAmount(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys: KeysCollection, crop_ranges: None) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)

        self.crop_ranges = crop_ranges

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key][:,
                     self.crop_ranges[0][0]:-self.crop_ranges[0][1],
                     self.crop_ranges[1][0]:-self.crop_ranges[1][1],
                     self.crop_ranges[2][0]:-self.crop_ranges[2][1]]

        return d
