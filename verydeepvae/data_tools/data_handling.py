import numpy as np
import h5py
import os
import shutil


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
