from torchio import Subject, ScalarImage
from monai.transforms.compose import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping
import numpy as np


class WrappedRandomGhosting(MapTransform):
    """
    trans: a torchio tranformation, e.g. RandomGhosting(p=1, num_ghosts=(4, 10))
    keys: a list of keys to apply the trans to, e.g. keys = ["img"]
    """

    def __init__(self, keys: KeysCollection, trans: None) -> None:
        """ """
        super().__init__(keys)

        self._seed = 0
        self.keys = keys
        self.trans = trans

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)

        for k in range(len(self.keys)):
            subject = Subject(img=ScalarImage(tensor=d[self.keys[k]]))
            if k == 0:
                transformed = self.trans
            else:
                transformed = transformed.get_composed_history()
            transformed = transformed(subject)
            d[self.keys[k]] = transformed["img"].data

        return d
