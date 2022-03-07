import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageOps


class JPEGDataset(Dataset):
    def __init__(self, jpeg_dir, file_names, csv_attr_path, transforms=None, attr_to_keep=[8, 9, 11, 20, 39],
                 hyper_params=None):

        self.jpeg_dir = jpeg_dir
        self.file_names = file_names
        self.metadata = csv_attr_path
        self.transforms = transforms
        self.attr_to_keep = attr_to_keep

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_id = self.file_names[index]
        file_path = os.path.join(self.jpeg_dir, file_id)
        img = Image.open(file_path)
        # img = img.rotate(90)

        try:
            img = ImageOps.grayscale(img)
        except:
            print("Error")
            print(file_path)
            quit()

        # # # Clamp
        # # min = np.percentile(img, 0.05)
        # # max = np.percentile(img, 99.5)
        # # img = np.clip(img, min, max)
        #
        # # Standardise
        # mean = np.mean(img)
        # std = np.std(img)
        # if std > 0:
        #     img = (img - mean) / std
        #
        # # Scale to [-1, 1]
        # mi = np.min(img)
        # ma = np.max(img)
        # img = (img - mi) / (ma - mi)
        # img = (2 * img) - 1
        #
        # # Turn back into a PIL so we can apply the transformations
        # img = Image.fromarray(img)
        # if self.transforms:
        #     img = self.transforms(img)

        img = np.array(img)
        img = np.expand_dims(img, 0)

        img = img.astype(np.float32)

        img = torch.from_numpy(img)

        if self.metadata is None:
            l = np.zeros_like(1)
        else:
            l = self.metadata.loc[self.metadata['image_id'] == file_id].to_numpy()[:, 1:].astype(np.int8)
            l = np.squeeze(l)
            l = (l + 1) // 2
            l = l[self.attr_to_keep]

        return img, l
