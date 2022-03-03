import os
import torch
import numpy as np
# from monai.data import NiftiDataset
from torch.utils.data import DataLoader
from . import data_handling
from torchvision import transforms
from PIL import Image


class DatasetCreator:
    """
    This class returns dataset and data loaders if a 3D data shape is specified in the hyperparameters, otherwise
    it loads the specified h5py dataset into RAM which can be iterated through using a python generator.
    # min, max = 0, 14364.451  # IXI
    # min, max = -3.38, 4136.74  # Biobank extracted
    """
    def __init__(self, **kwargs):
        super().__init__()

        if 'h5py_filename' in kwargs["hyper_params"] and 'nifti_dir' in kwargs["hyper_params"]:
            print("Specify EITHER an h5py filename OR a nifti directory.")
            quit()

        elif 'jpeg_dir' in kwargs["hyper_params"]:
            print("JPEG directory specified in hyperparameters")
            print("JPEG base dir: " + kwargs["hyper_params"]['jpeg_dir'])
            from jpeg_dataset import JPEGDataset

            jpeg_dir = kwargs["hyper_params"]['jpeg_dir']
            file_names = [f for f in os.listdir(jpeg_dir) if os.path.isfile(os.path.join(jpeg_dir, f))]
            csv_attr_path = None

            if 'max_jpegs_to_use' in kwargs["hyper_params"]:
                file_names = file_names[0:kwargs["hyper_params"]['max_jpegs_to_use']]

            train_frac = kwargs["hyper_params"]['train_frac']
            print("Creating training/validation set split: proportion for training: " + str(train_frac))
            cardinality = len(file_names)
            cardinality_train = int(train_frac * cardinality)
            file_names_train = file_names[0:cardinality_train]
            file_names_val = file_names[cardinality_train:]

            train_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05,
                                                                          hue=0.1),
                                                   transforms.RandomAffine(3, translate=(0, 0.1), scale=(0.75, 1),
                                                                           shear=3, resample=Image.BILINEAR,
                                                                           fillcolor=0),
                                                   transforms.CenterCrop(
                                                       kwargs["hyper_params"]['jpeg_target_shape'][0]),
                                                   transforms.RandomVerticalFlip(p=0.5),
                                                   transforms.ToTensor()])
            val_transforms = transforms.Compose([transforms.CenterCrop(kwargs["hyper_params"]['jpeg_target_shape'][0]),
                                                 transforms.ToTensor()])

            self.dataset_train = JPEGDataset(jpeg_dir, file_names_train, csv_attr_path, train_transforms)
            self.dataset_val = JPEGDataset(jpeg_dir, file_names_val, csv_attr_path, val_transforms)

            self.loader_train = DataLoader(self.dataset_train, batch_size=kwargs["hyper_params"]['batch_size'],
                                           shuffle=True,
                                           num_workers=kwargs["hyper_params"]['number_of_workers'],
                                           pin_memory=torch.cuda.is_available())
            self.loader_val = DataLoader(self.dataset_val, batch_size=kwargs["hyper_params"]['batch_size'],
                                         shuffle=False,
                                         num_workers=kwargs["hyper_params"]['number_of_workers'],
                                         pin_memory=torch.cuda.is_available())

            # self.loader_val = self.loader_train

            self.cardinality_train = len(self.dataset_train)
            self.cardinality_val = len(self.dataset_val)

            self.is_3d = False
            # self.is_colour = True
            # self.data_shape = [3] + kwargs["hyper_params"]['jpeg_target_shape'] * 2
            self.data_shape = [1] + kwargs["hyper_params"]['jpeg_target_shape'] * 2
            self.is_colour = False

            self.sampler_train = None
            self.sampler_val = None

        elif 'nifti_dir' in kwargs["hyper_params"]:
            print("Nifti directory specified in hyperparameters")
            print("Nifti base dir: " + kwargs["hyper_params"]['nifti_base_dir'])
            print("Nifti data: " + kwargs["hyper_params"]['nifti_dir'])
            nifti_dir = os.path.join(kwargs["hyper_params"]['nifti_base_dir'], kwargs["hyper_params"]['nifti_dir'])
            print("Full nifti path: " + nifti_dir)
            filenames = os.listdir(nifti_dir)
            nifti_filenames = []
            for file in filenames:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    nifti_filenames.append(file)

            print("Number of niftis: " + str(len(nifti_filenames)))
            nifti_paths = [os.path.join(kwargs["hyper_params"]['nifti_base_dir'],
                                        kwargs["hyper_params"]['nifti_dir'],
                                        name) for name in nifti_filenames]
            labels = np.zeros([len(nifti_paths)], dtype=np.int64)
            training_set_size = np.floor(len(nifti_paths) * kwargs["hyper_params"]['train_frac']).astype(np.int32)

            if 'nii_target_shape' in kwargs["hyper_params"]:
                self.data_shape = kwargs["hyper_params"]['nii_target_shape']
            else:
                print("You must specify the resample target shape using the data_shape key!")
                quit()

            train_transforms = [monai_trans.LoadImaged(keys=keys),
                                monai_trans.AddChanneld(keys=keys),
                                monai_trans.Resized(keys=keys, spatial_size=tuple(hyper_params['nii_target_shape']))]

            train_transforms += [ClampByPercentile(keys=["full_brain", "partial_brain"], lower=0.05, upper=99.5)]
            train_transforms += [monai_trans.NormalizeIntensityd(keys=["full_brain", "partial_brain"])]
            train_transforms += [monai_trans.ScaleIntensityd(["full_brain", "partial_brain"], minv=-1.0, maxv=1.0)]

            train_transforms = monai_trans.Compose(train_transforms)
            val_transforms = train_transforms

            self.dataset_train = NiftiDataset(image_files=nifti_paths[0:training_set_size],
                                              labels=labels[0:training_set_size],
                                              transform=train_transforms)
            self.dataset_val = NiftiDataset(image_files=nifti_paths[training_set_size::],
                                            labels=labels[training_set_size::],
                                            transform=val_transforms)

            # pin_memory = torch.cuda.is_available()
            # pin_memory = True
            pin_memory = False
            self.loader_train = DataLoader(self.dataset_train,
                                           batch_size=kwargs["hyper_params"]['batch_size'],
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=kwargs["hyper_params"]['number_of_workers'],
                                           pin_memory=pin_memory)
            self.loader_val = DataLoader(self.dataset_val,
                                         batch_size=kwargs["hyper_params"]['batch_size'],
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=kwargs["hyper_params"]['number_of_workers'],
                                         pin_memory=pin_memory)

            self.cardinality_train = len(self.dataset_train)
            self.cardinality_val = len(self.dataset_val)
            
            self.is_3d = True
            self.is_colour = False

        elif 'h5py_filename' in kwargs["hyper_params"]:
            print("H5py filename specified in hyperparameters")
            [self.cardinality_train, self.dataset_train, self.cardinality_val, self.dataset_val, self.data_shape,
             self.is_3d, self.is_colour] = data_handling.h5pyDataset(kwargs["hyper_params"])

        else:
            print("Specify an h5py filename or a nifti directory.")
            quit()

        self.batch_count_train = np.ceil(self.cardinality_train / kwargs["hyper_params"]['batch_size']).astype(np.int32)
        self.batch_count_val = np.ceil(self.cardinality_val / kwargs["hyper_params"]['batch_size']).astype(np.int32)

        print("Training set size: " + str(self.cardinality_train))
        print("Validation set size: " + str(self.cardinality_val))
        print("Training batches per epoch: " + str(self.batch_count_train))
        print("Validation batches per epoch: " + str(self.batch_count_val))

    def data_loader(self, cardinality, dataset, batch_size):
        data = dataset[0]
        labels = dataset[1]
        num = 0

        while num < cardinality:

            if num < cardinality - batch_size:
                right_bound = num + batch_size
            else:
                right_bound = num + cardinality - num

            yield [data[num:right_bound], labels[num:right_bound]]

            num += batch_size
