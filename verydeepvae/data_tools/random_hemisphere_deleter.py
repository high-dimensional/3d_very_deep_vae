from typing import Any, Dict, Hashable, Mapping, Optional
import numpy as np
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable


class HemisphereDeleter(Randomizable, MapTransform):
    """
    """

    def __init__(
        self,
        keys: KeysCollection,
        halfway_point=None,
        wobble_range=None,
        midline_correction=None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            range:

        """
        super().__init__(keys)

        self.halfway_point = halfway_point
        self.wobble_range = wobble_range
        self.midline_correction = midline_correction

    def randomize(self, data: Optional[Any] = None) -> None:
        self.side = self.R.randint(low=0, high=2)  # high not inclusive
        self.wobble = self.R.randint(low=-self.wobble_range, high=self.wobble_range + 1)
        # self.midline_correction = {'right': -1, 'left': 0}

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()

        for key in self.keys:

            mask = np.ones_like(d[key])

            if self.side == 0:
                mask[
                    0,
                    0 : self.halfway_point
                    + self.wobble
                    + self.midline_correction["right"],
                    :,
                    :,
                ] = 0
            elif self.side == 1:
                mask[
                    0,
                    self.halfway_point
                    + self.wobble
                    + self.midline_correction["left"] :,
                    :,
                    :,
                ] = 0
            else:
                quit()

            fill_val = d[key][0, 0, 0, 0]
            d[key] = d[key] * mask + fill_val * (1 - mask)
            d[key + "_mask"] = mask

        return d
