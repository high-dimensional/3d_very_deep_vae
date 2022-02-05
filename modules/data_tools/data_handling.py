import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import matplotlib as mpl
mpl.use("AGG")
import matplotlib.pyplot as plt
import h5py
import sys
import math
import os
import misc

"""
These are tools related to loading, formatting and interrogating data on disk
"""

def delete_directory_contents(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to empty directory %s. Reason: %s' % (file_path, e))


def save_to_mat(dictionary, filepath):
    import scipy.io as sio
    file_handler = open(filepath, "wb")
    sio.savemat(file_handler, mdict=dictionary, do_compression=False)
    file_handler.close()


def hdf5_info(file_path):
    f = h5py.File(file_path, 'r')
    fields = list(f.keys())
    raw_data = f[fields[0]]
    data_shape = raw_data.shape[0:-1]
    data_type = raw_data.dtype
    length = raw_data.shape[-1]
    f.close()
    output = [file_path, length, data_shape, data_type, fields]
    print("hdf5 info: ", output)
    return output


def compute_min_max_from_loader(data_loader):
    count = 0
    for batch in data_loader:
        batch_features = batch[0].cpu().detach().numpy()
        max_new = np.max(batch_features)
        min_new = np.min(batch_features)

        if count == 0:
            max = max_new
            min = min_new
        else:
            if max_new > max:
                max = max_new
            if min_new < min:
                min = min_new
        count += 1
        progress_bar(count, len(data_loader), prefix='Computing min/max:')

    print("Min/max: {:.2f}/{:.2f}".format(min, max))
    return min, max


def h5pyDataset(hyper_params, drop_remainder=False):
    """
    Create a dataset from an h5py file
    """
    print("Loading h5py dataset: " + hyper_params['h5py_filename'])

    batch_size = hyper_params['batch_size']
    train_frac = hyper_params['train_frac']
    full_path = hyper_params['h5py_base_dir'] + hyper_params['h5py_filename']

    examples = None
    labels = None
    label_name = None
    hdf5_original_data = h5py.File(full_path, 'r')

    # Now we work out what to use as data and what to use as labels
    if 'h5py_data_variable_name' in hyper_params:
        variable_name = hyper_params['h5py_data_variable_name']
    else:
        print("No 'h5py_data_variable_name' specified: guessing which arrays to use for data and labels")
        # If the array isn't named, find all the names
        num_vars, names, shapes = misc.iterate_through_h5py(hdf5_original_data)
        if num_vars == 0:
            print("No arrays found in file - quitting.")
            quit()
        elif num_vars == 1:
            print("Only one array in hdf5 file: loading '" + names[0] + "'")
            variable_name = names[0]
        elif num_vars == 2:
            dims = [np.prod(shape) for shape in shapes]
            print("Two arrays found, assuming bigger one is data smaller is labels")
            variable_name = 'slice_array_padded'
        elif num_vars > 2:
            print("More than 2 arrays found")
            print("Assuming biggest one is data")
            dims = [np.prod(shape) for shape in shapes]
            sort_index = np.argsort(dims, )
            variable_name = names[sort_index[-1]]
            label_name = names[sort_index[-2]]

            if 'labels' in names:
                print("Assuming array named 'labels' is the labels array")
                # idx = names.index('labels')
                label_name = 'labels'
            else:
                print("Assuming second biggest is labels")

    print("Label array: '" + str(label_name) + "'")
    print("Data array: '" + str(variable_name) + "'")
    hdf5_original_data[variable_name].astype(np.float32)
    data = np.array(hdf5_original_data[variable_name])
    hdf5_array_shape = list(data.shape)
    numbered_dims_reversed = list(range(len(data.shape)))[::-1]
    examples = np.transpose(data, numbered_dims_reversed).astype(np.float32)

    if 'hdf5_examples_to_use' in hyper_params:
        # Should shuffle first
        examples = examples[0:hyper_params['hdf5_examples_to_use']]

    cardinality = examples.shape[0]
    print("Shape of data array: " + str(data.shape))

    if label_name is not None:
        hdf5_original_data[label_name].astype(np.float32)
        labels = np.array(hdf5_original_data[label_name])
        labels = np.transpose(labels, [1, 0]).astype(np.float32)

    # We need to detect if there's a colour channel and put it in the right place
    # And also decide here whether to insert a singleton colour channel because PyTorch want B x C x H x W
    is_3d = False
    is_colour = False
    if len(numbered_dims_reversed) == 2:
        pass
    elif len(numbered_dims_reversed) == 3:
        pass
    elif len(numbered_dims_reversed) == 4:
        print("4 dimensions detected")
        if 'h5py_data_is_3d' in hyper_params:
            if hyper_params['h5py_data_is_3d']:
                is_3d = True
            else:
                is_3d = False
        else:
            print("`h5py_data_is_3d' not specified")
            if hdf5_array_shape[0] == 3:
                print("Detected a colour channel, assuming data is 2D colour images")
                examples = np.transpose(examples, [0, 3, 1, 2])
                is_3d = False
                is_colour = True
            elif hdf5_array_shape[0] == 1:
                print("Detected a greyscale channel, assuming data is 2D greyscale images")
                is_3d = False
            else:
                print("No colour or a greyscale channel found, assuming data is volumetric greyscale")
                is_3d = True
    elif len(numbered_dims_reversed) > 4:
        print("More than 4 dimensions: quitting")
        quit()

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Overriding hdf5 loading code")
    # examples = np.transpose(data, [0, 2, 3, 1]).astype(np.float32)
    examples = data.astype(np.float32)
    cardinality = examples.shape[0]
    is_3d = False
    is_colour = True

    data_shape = examples.shape[1:]

    if hyper_params['shuffle_hdf5_data']:
        print("Data shuffled")
        random_index = np.random.permutation(examples.shape[0]).astype(np.int32)
        examples = examples[random_index]

    if train_frac < 1:
        print("Creating training/validation set split: proportion for training: " + str(train_frac))
        cardinality_train = int(train_frac * cardinality)
        cardinality_val = cardinality - cardinality_train
        examples_train = examples[0:cardinality_train]
        examples_val = examples[cardinality_train:]

    if labels is None:
        if train_frac < 1:
            labels_train = np.zeros_like(examples_train)
            labels_val = np.zeros_like(examples_val)
        else:
            labels = np.zeros_like(examples)
    else:
        if train_frac < 1:
            labels_train = labels[0:cardinality_train]
            labels_val = labels[cardinality_train:]

    if drop_remainder:
        if train_frac < 1:
            cardinality_train = cardinality_train // batch_size
            cardinality_train = cardinality_train * batch_size
            examples_train = examples_train[0:cardinality_train]

            cardinality_val = cardinality_val // batch_size
            cardinality_val = cardinality_val * batch_size
            examples_val = examples_val[0:cardinality]
        else:
            cardinality = cardinality // batch_size
            cardinality = cardinality * batch_size
            examples = examples[0:cardinality]

    print("-> Replacing labels with zero vector")

    if train_frac < 1:
        if not is_colour:
            examples_train = np.expand_dims(examples_train, 1)
            examples_val = np.expand_dims(examples_val, 1)

        examples_train = torch.Tensor(examples_train)
        examples_val = torch.Tensor(examples_val)
        labels_train = torch.Tensor(labels_train)
        labels_val = torch.Tensor(labels_val)
    else:
        if not is_colour:
            examples = np.expand_dims(examples, 1)

        examples = torch.Tensor(examples)
        labels = torch.Tensor(labels)

    if train_frac < 1:
        return [cardinality_train, [examples_train, labels_train],
                cardinality_val, [examples_val, labels_val], data_shape, is_3d, is_colour]
    else:
        return cardinality, [examples, labels], data_shape, is_3d, is_colour
