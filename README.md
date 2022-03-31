# `3d_very_deep_vae`

[PyTorch](https://pytorch.org/) implementations of [variational autoencoder models](https://en.wikipedia.org/wiki/Variational_autoencoder) for generating synthetic three-dimensional images based on neuroimaging training data.

## Installation

To install the `verydeepvae` package a 64-bit Linux or Windows system with one of [Python 3.7, 3.8 or 3.9](https://www.python.org/downloads/) is required and version 19.3 or above of the [Python package installer `pip`](https://pip.pypa.io/en/stable/getting-started/#ensure-you-have-a-working-pip). A local installation of version 11.3 or above of the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and a compatible graphic processing unit (GPU) and associated driver will also be required to run the code.

Platform-dependent requirements files specifying all Python dependencies with pinned versions are provided in the [`requirements`](requirements) directory with the naming scheme `{python_version}-{os}-requirements.txt` where `{python_version}` is one of `py37`, `py38` and `py39` and `{os}` one of `linux` and `windows`. The Python dependencies can be installed in to the current Python environment using `pip` by running

```console
pip install -r requirements/{python_version}-{os}-requirements.txt
```

from a local clone of this repository, where `{python_version}` and `{os}` are the appropriate values for the Python version and operating system of the environment being installed in.

Once the Python dependencies have been installed the `verydeepvae` package can be installed by running

```console
pip install .
```
from the root of the repository.

## Input data

The code is currently designed to train variational autoencoder models on volumetric neuroimaging data from the [UK Biobank imaging study](https://www.ukbiobank.ac.uk/enable-your-research/about-our-data/imaging-data). This dataset is not publicly accessible and requires [applying for access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). The package requires the imaging data to be accessible on the node(s) used for training as a flat directory of [NiFTI](https://radiopaedia.org/articles/nifti-file-format?lang=gb) files of [_fluid-attenuated inverse recovery_ (FLAIR)](https://radiopaedia.org/articles/fluid-attenuated-inversion-recovery?lang=gb) images. The FLAIR images are expected to be affine-aligned to a template and skull-stripped using the [_Statistical Parameter Mapping_ (SPM)](https://www.fil.ion.ucl.ac.uk/spm/) software package. An alternative openly accessible dataset will be provided at a later date.


## Model training

Scripts are included for training variational autoencoder models with three pre-defined configurations on UK Biobank FLAIR image data. The three pre-defined model configurations in the `VeryDeepVAE_32x32x32`, `VeryDeepVAE_64x64x64` and `VeryDeepVAE_128x128x128` directories differ only in the target resolution of the generated images (respectively `32×32×32`, `64×64×64` and `128×128×128`) and the number of layers in the autoencoder model (see [_Layer definitions_](#layer-definitions) below), with the `64×64×64` configuration having one more layer than the `32×32×32` configuration and the `128×128×128` configuration having one more layer again than the `64×64×64` configuration.

The model configuration defined in `VeryDeepVAE_128x128x128` has a peak GPU memory usage of 31.9GB, so should be runnable on a GPU with 32GB of device memory. Changing the latent dimensionality per channel from the default 7 to 6 by setting the `latents_per_channel` hyperparameter should make it fit comfortably in 32GB if that becomes a problem.

To create a new model configuration create a new folder in the root of the local clone of the repository and create a `run_autoencoder.py` file in it, following the structure of the included examples to define a dictionary `hyper_params` containing the hyperparameter values specifying the model and training configuration. See the [_Hyperparameters_ section below](#hyperparameters) for details of some of the more important properties.

### Example usages

In the below `{configuration_directory}` should be replaced with the relevant directory for the desired model configuration to train (for example `VeryDeepVAE_128x128x128`) and `{nifti_flair_directory}` with the path to the directory containing the UK Biobank imaging data NiFTI FLAIR image files. In all cases it is assumed the commands are being executed in a Unix shell such as `sh` or `bash` - if using an alternative command-line interpreter such as `cmd.exe` or PowerShell on Windows the commands will not work.

#### Running on a single GPU

To run on one GPU (see the [point below about `CUDA_devices` hyperparameter](#cuda-devices)):

```sh 
python {configuration_directory}/run_autoencoder.py \
  --CUDA_devices 0 --local_rank 0 --nifti_flair_dir {nifti_flair_directory} 
```
  
#### Running on multiple GPUs

To run on a single node with 8 GPU devices:

```sh
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  {configuration_directory}/run_autoencoder.py \ 
  --CUDA_devices 0,1,2,3,4,5,6,7 --nifti_flair_dir {nifti_flair_directory}
```

To specify the backend and endpoint:

```sh
python -m torch.distributed.run \ 
  --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint={endpoint} \
  {configuration_directory}/run_autoencoder.py \ 
  --CUDA_devices 0,1,2,3,4,5,6,7 --nifti_flair_dir {nifti_flair_directory}
```
where `{endpoint}` is the endpoint where the rendezvous backend is running in the form `host_ip:port`.

#### Running on multiple nodes

To run on two nodes, each with 8 GPU devices:

_On first node_

```sh
python -m torch.distributed.run \ 
  --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr={ip_address} --master_port={port_number} \
  {configuration_directory}/run_autoencoder.py \
  --CUDA_devices 0,1,2,3,4,5,6,7 --nifti_flair_dir {nifti_flair_directory}
```
_On second node_

```sh
python -m torch.distributed.run \ 
  --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr={ip_address} --master_port={port_number} \
  {configuration_directory}/run_autoencoder.py \
  --CUDA_devices 0,1,2,3,4,5,6,7 --nifti_flair_dir {nifti_flair_directory}
```

where `{ip_address}` is the IP address of the rank 0 node and `{port_number}` is a free 
port on the rank 0 node. 

#### Halting zombie runs

To kill zombies run

```sh
kill $(ps aux | grep config.py | grep -v grep | awk '{print $2}') and kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')
```

## Hyperparameters

The hyperparameters specifying the model configuration are specified in a dictionary `hyper_params` constructed in the `run_autoencoder.py` scripts, with some hyperparameters also possible to set instead via command line arguments. Details of some of the more important hyperparameters and their keys in the `hyper_params` dictionary:

- `nifti_flair_dir` : 
  All the Biobank NiFTIs should be in this directory. This should be specifed _either_ as a string with key `nifti_flair_dir` in the hyperparameters dictionary or as a command line argument `--nifti_flair_dir` to the `run_autoencoders.py` script.
- `batch_size`:
  The number of images per minibatch for the stochastic gradient descent training algorithm. For the `128×128×128` configuration the model has been successfully trained with a batch size of 2 or 3 on _Cambridge-1_, and a batch size of 1 on V100 DGX1s. Much higher batch sizes are possible at lower resolutions.
- <a href='#' id='cuda-devices'></a>`CUDA_devices`:
  Zero-indexed integer indices of the CUDA GPU devices to use in training on the current node. This should either be specified as a list of strings, for example `['0']` or `['0', '1']`, with key `CUDA_devices` in the hyperparameters dictionary or as a command line argument `--CUDA_devices` as comma-separated integers, for example `--CUDA_devices 0` or `--CUDA_devices 0,1`.
- `max_niis_to_use`:
  Use this to define a shorter epoch, for example to quickly test visualisations are being saved correctly.
- `nii_target_shape`:
  Specifies the target resolution to generate images at as a list of three positive integers corresponding to integer powers of 2, for example `[128, 128, 128]` for a `128×128×128` resolution.
- `visualise_training_pipeline_before_starting`:
  Set this to `True` to see a folder (`pipeline_test`, in the output folder) of augmented examples.

### Layer definitions

The model architecture is specified by a series of hyperparamters `latents_per_channel`, `channels_per_latent`, `channels`, `kernel_sizes_bottom_up` and `kernel_sizes_top_down`, each of which is list of `k + 1` integers where `k` is the base-2 logarithm of the resolution along each dimension - for example for the `128×128×128` configuration `k = 7`. The corresponding entries in all lists define a convolution block and after each of these we downsample by a factor of two in each spatial dimension on the way up and upsample by a factor of two in each spatial dimension on the way back down. This version of the code has not been tested when these lists have fewer than `k + 1` elements - you have been warned!

An example valid definition for the `128×128×128` configuration is

```Python
hyper_params['latents_per_channel'] = [2, 7, 6, 5, 4, 3, 2, 1]
hyper_params['channels_per_latent'] = [20, 20, 20, 20, 20, 20, 20, 200]  
hyper_params['channels'] = [20, 40, 60, 80, 100, 120, 140, 160]
hyper_params['kernel_sizes_bottom_up'] = [3, 3, 3, 3, 3, 3, 2, 1]
hyper_params['kernel_sizes_top_down'] = [3, 3, 3, 3, 3, 3, 2, 1]  
```

where for example the first line specifies that, reading left to right, we have 2 latent dimensions per channel at `128×128×128` resolution, 7 latent dimensions per channel at `64×64×64` resolution, 6 latents per channel at `32×32×32` resolution, 5 latent dimensions per channel at `16×16×16` resolution and so on.

## Licence

The code is under the [GNU General Public License Version 3](LICENSE).