from torchio import Subject, LabelMap, ScalarImage, RandomAffine
from torchio import transforms as torchio_trans
from monai.transforms.compose import MapTransform, Randomizable
from monai.config import KeysCollection
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torchio as tio
from monai.transforms import LoadNifti, Randomizable, apply_transform
from monai import transforms as monai_trans
from numpy import random
import torch
import numpy
import torchio


class WrappedRandomGhosting(MapTransform):
    """
    trans: a torchio tranformation, e.g. RandomGhosting(p=1, num_ghosts=(4, 10))
    keys: a list of keys to apply the trans to, e.g. keys = ["img"]
    """

    def __init__(
        self,
        keys: KeysCollection,
            trans: None
    ) -> None:
        """
        """
        super().__init__(keys)

        self._seed = 0
        self.keys = keys
        self.trans = trans

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d = dict(data)

        for k in range(len(self.keys)):
            subject = Subject(img=ScalarImage(tensor=d[self.keys[k]]))
            if k == 0:
                transformed = self.trans
            else:
                transformed = transformed.get_composed_history()
            transformed = transformed(subject)
            d[self.keys[k]] = transformed['img'].data

        return d