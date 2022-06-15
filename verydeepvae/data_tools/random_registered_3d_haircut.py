from typing import Any, Dict, Hashable, Mapping, Optional
import numpy as np
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable


class ThreeDHaircut(Randomizable, MapTransform):
    """ """

    def __init__(
        self, keys: KeysCollection, range, axis: int = 2, prob: float = 0.1
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            range:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)

        """
        super().__init__(keys)

        self.range = range
        self.prob = prob
        self.axis = axis
        self._do_transform = False

    def randomize(self, data: Optional[Any] = None) -> None:
        self.index_1 = int(self.R.uniform(low=1, high=self.range[0]))
        self.index_2 = int(self.R.uniform(low=1, high=self.range[1]))
        self.index_3 = int(self.R.uniform(low=1, high=self.range[2]))
        self._do_transform = self.R.random() < self.prob
        self.pick_axis = self.R.random()

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.keys:
            mask = np.zeros_like(d[key])

            if 0 <= self.pick_axis < 0.333:
                mask[0, self.index_1 : -self.index_1, :, :] = 1
            elif 0.333 <= self.pick_axis < 0.666:
                mask[0, :, self.index_2 : -self.index_2, :] = 1
            else:
                mask[0, :, :, self.index_3 : -self.index_3] = 1

            d[key] = d[key] * mask

        return d
