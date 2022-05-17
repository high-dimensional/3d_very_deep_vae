from typing import Any, Dict, Hashable, Mapping, Optional
import numpy as np
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.spatial.array import Resize


class Anisotropiser(Randomizable, MapTransform):
    """
    """

    def __init__(
        self,
        keys: KeysCollection,
        scale_range,
        prob: float = 0.1,
        allow_missing_keys: bool = True,
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

        self.scale_range = scale_range
        self.prob = prob
        self._do_transform = False
        self.allow_missing_keys = allow_missing_keys

    def randomize(self, data: Optional[Any] = None) -> None:
        self.axis = self.R.randint(low=0, high=3)
        self.scale = self.R.uniform(low=self.scale_range[0], high=self.scale_range[1])
        self._do_transform = self.R.random() < self.prob

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.keys:

            if key in d:
                shape_old = d[key].shape[1:]

                # For the target shape, shrink along one and only one dimension
                shape_new = [-1, -1, -1]
                shape_new[self.axis] = int(self.scale * shape_old[self.axis])

                self.resizer_old = Resize(spatial_size=tuple(shape_old))
                self.resizer_new = Resize(spatial_size=tuple(shape_new))

                d[key] = self.resizer_old(self.resizer_new(d[key]))
            else:
                if not self.allow_missing_keys:
                    print("key not found in Anisotropiser. Exiting.")
                    quit()

        return d
