from typing import Dict, Hashable, Mapping
from monai.config import NdarrayTensor
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform


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

    def __call__(
        self, data: Mapping[Hashable, NdarrayTensor]
    ) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = d[key][
                :,
                self.crop_ranges[0][0] : -self.crop_ranges[0][1],
                self.crop_ranges[1][0] : -self.crop_ranges[1][1],
                self.crop_ranges[2][0] : -self.crop_ranges[2][1],
            ]

        return d
