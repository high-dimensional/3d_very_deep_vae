from typing import Any, Callable, Optional, Sequence
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Randomizable, apply_transform
from monai.utils import get_seed
import torch
from monai import transforms as monai_trans
import h5py


class HDF5Loader(Dataset):
    """
    """

    def __init__(
            self,
            hdf5_path: str,
            train_frac: float = 1.0,
            training_set: bool = True,
            shared_transforms: Optional[Callable] = None,
            standardise: bool = False,
            hyper_params: dict = None
    ) -> None:

        self.hdf5_path = hdf5_path
        self.train_frac = train_frac
        self.training_set = training_set
        self.shared_transforms = shared_transforms
        self.standardise = standardise
        self.hyper_params = hyper_params
        self._seed = 0  # transform synchronization seed

        hdf5_original_data = h5py.File(self.hdf5_path, 'r')
        hdf5_original_data['data'].astype(np.float32)
        self.data = np.array(hdf5_original_data['data'])
        numbered_dims_reversed = list(range(len(self.data.shape)))[::-1]
        self.data = np.transpose(self.data, numbered_dims_reversed).astype(np.float32)

        self.original_cardinality = int(self.data.shape[0])
        self.original_data_shape = self.data.shape[2:]

        if self.train_frac < 1:
            left_of_split = int(np.floor(self.train_frac * self.original_cardinality))
            right_of_split = int(self.original_cardinality - left_of_split)

            if training_set:
                self.data = self.data[0:left_of_split]
                self.cardinality = left_of_split
            else:
                self.data = self.data[left_of_split:]
                self.cardinality = right_of_split

        if self.original_data_shape == self.hyper_params['nii_target_shape']:
            self.initial_trans = monai_trans.Compose([monai_trans.AddChannel()])
        else:
            self.initial_trans = \
                monai_trans.Compose([monai_trans.AddChannel(),
                                     monai_trans.Resize(tuple(self.hyper_params['nii_target_shape']))])

        if self.shared_transforms is None:
            self.final_trans = monai_trans.ToTensor()
        else:
            self.final_trans = \
                monai_trans.Compose([monai_trans.Resize(tuple(self.hyper_params['nii_target_shape'])),
                                     monai_trans.ToTensor()])

    def __len__(self) -> int:
        return self.cardinality

    def __getitem__(self, index: int):

        def standardise(img):
            m, s = torch.mean(img), torch.std(img)
            s[s < 1e-6] = 1
            return (img - m) / s

        t1 = self.data[index, 0, :, :, :]
        flair = self.data[index, 1, :, :, :]
        b0 = self.data[index, 2, :, :, :]
        b1000 = self.data[index, 3, :, :, :]

        (t1, flair, b0, b1000) = apply_transform(self.initial_trans, (t1, flair, b0, b1000), map_items=True)

        if self.shared_transforms is not None:
            if isinstance(self.shared_transforms, Randomizable):
                self.shared_transforms.set_random_state(seed=self._seed)
            t1 = apply_transform(self.shared_transforms, t1)

            if isinstance(self.shared_transforms, Randomizable):
                self.shared_transforms.set_random_state(seed=self._seed)
            flair = apply_transform(self.shared_transforms, flair)

            if isinstance(self.shared_transforms, Randomizable):
                self.shared_transforms.set_random_state(seed=self._seed)
            b0 = apply_transform(self.shared_transforms, b0)

            if b1000 is not None:
                if isinstance(self.shared_transforms, Randomizable):
                    self.shared_transforms.set_random_state(seed=self._seed)
                b1000 = apply_transform(self.shared_transforms, b1000)

        (t1, flair, b0, b1000) = apply_transform(self.final_trans, (t1, flair, b0, b1000), map_items=True)

        data = [t1, flair, b0, b1000]

        if self.standardise:
            data = [standardise(d) for d in data]

        return tuple(data)
