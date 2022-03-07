from collections.abc import Iterable
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.utils import dtype_torch_to_numpy, ensure_tuple_size


class ClampByPercentile(MapTransform):
    """
    Clamp between the lower and upper percentiles, e.g. lower = 1, upper = 99 to remove the lowest and highest 1% of
    the signal.
    """

    def __init__(self,
                 keys: KeysCollection,
                 lower: None,
                 upper: None,
                 allow_missing_keys = True) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            range:

        """
        super().__init__(keys)

        self.lower = lower
        self.upper = upper
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        for key in self.keys:

            if key in d:
                min = np.percentile(d[key], self.lower)
                max = np.percentile(d[key], self.upper)
                d[key] = np.clip(d[key], min, max)
            else:
                if not self.allow_missing_keys:
                    print("key not found in ClampByPercentile. Exiting.")
                    quit()

        return d
