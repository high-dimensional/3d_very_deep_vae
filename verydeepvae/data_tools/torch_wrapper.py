from torchio import Subject, LabelMap, ScalarImage
from monai.transforms.compose import MapTransform, Randomizable
from monai.config import KeysCollection
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import torchio as tio
import torchio
import numpy as np


class TorchIOWrapper(Randomizable, MapTransform):
    """
    Use torchio transformations in Monai and control which dictionary entries are transformed in synchrony!
    trans: a torchio tranformation, e.g. RandomGhosting(p=1, num_ghosts=(4, 10))
    keys: a list of keys to apply the trans to, e.g. keys = ["img"]
    p: probability that this trans will be applied (to all the keys listed in 'keys')
    """

    def __init__(
        self,
        keys: KeysCollection,
            trans: None,
            p: None
    ) -> None:
        """
        """
        super().__init__(keys)

        self.keys = keys
        self.trans = trans
        self.prob = p
        self._do_transform = False

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:

        d = dict(data)

        self.randomize()
        if not self._do_transform:
            return d

        for k in range(len(self.keys)):
            subject = Subject(datum=ScalarImage(tensor=d[self.keys[k]]))
            if k == 0:
                transformed = self.trans
            else:
                transformed = transformed.get_composed_history()
            transformed = transformed(subject)
            d[self.keys[k]] = transformed['datum'].data

        return d
