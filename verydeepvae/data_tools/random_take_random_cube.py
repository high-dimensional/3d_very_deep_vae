from typing import Any, Dict, Hashable, Mapping, Optional
import numpy as np
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable


class TakeRandomCube(Randomizable, MapTransform):
    """ """

    def __init__(
        self,
        keys: KeysCollection,
        vol_shape=[64, 64, 64],
        min_cube_size=[64, 64, 64],
        max_cube_size=[64, 64, 64],
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`

        """
        super().__init__(keys)

        self.vol_shape = vol_shape
        self.min_cube_size = min_cube_size
        self.max_cube_size = max_cube_size

    def randomize(self, data: Optional[Any] = None) -> None:
        self.side_lengths = [
            self.R.randint(low=self.min_cube_size[0], high=self.max_cube_size[0]),
            self.R.randint(low=self.min_cube_size[1], high=self.max_cube_size[1]),
            self.R.randint(low=self.min_cube_size[2], high=self.max_cube_size[2]),
        ]

        self.offsets = [
            self.R.randint(low=0, high=self.vol_shape[0] - self.side_lengths[0]),
            self.R.randint(low=0, high=self.vol_shape[1] - self.side_lengths[1]),
            self.R.randint(low=0, high=self.vol_shape[2] - self.side_lengths[2]),
        ]

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()

        for key in self.keys:

            mask = np.ones_like(d[key])

            mask[
                0,
                self.offsets[0] : self.offsets[0] + self.side_lengths[0],
                self.offsets[1] : self.offsets[1] + self.side_lengths[1],
                self.offsets[2] : self.offsets[2] + self.side_lengths[2],
            ] = 0

            d[key] = d[key] * mask

        return d
