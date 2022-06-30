# `3d_very_deep_vae`

[![Continuous integration](https://github.com/high-dimensional/3d_very_deep_vae/actions/workflows/ci.yml/badge.svg)](https://github.com/high-dimensional/3d_very_deep_vae/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6782948.svg)](https://doi.org/10.5281/zenodo.6782948)

[PyTorch](https://pytorch.org/) implementation of (a streamlined version of) Rewon Child's 'very deep' variational autoencoder [(Child, R., 2021)](#child2021very) for generating synthetic three-dimensional images based on neuroimaging training data. The Wikipedia page for [variational autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder) contains some background material.

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

The code is currently designed to train variational autoencoder models on volumetric neuroimaging data from the [UK Biobank imaging study](https://www.ukbiobank.ac.uk/enable-your-research/about-our-data/imaging-data). This dataset is not publicly accessible and requires [applying for access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). The package requires the imaging data to be accessible on the node(s) used for training as a flat directory of [NIfTI](https://radiopaedia.org/articles/nifti-file-format?lang=gb) files of [_fluid-attenuated inverse recovery_ (FLAIR)](https://radiopaedia.org/articles/fluid-attenuated-inversion-recovery?lang=gb) images. The FLAIR images are expected to be affine-aligned to a template and skull-stripped using the [_Statistical Parameter Mapping_ (SPM)](https://www.fil.ion.ucl.ac.uk/spm/) software package.

As an alternative for testing purposes, a script [`generate_synthetic_data.py`](scripts/generate_synthetic_data.py) is included in the `scripts` directory which can be used to generate a set of NIfTI volumetric image files of a specified resolution. The generated volumetric images consist of randomly oriented and sized ellipsoid inclusions overlaid with Gaussian filtered background noise. The script allows specifying the number of files to generate, their resolution and parameters controlling the noise amplitude and length scale, and difference between ellipsoid inclusion and background.

To see the full set of command line arguments that can be passed to the training script run

```bash
python scripts/generate_synthetic_data.py --help
```

For example to generate a set of 10&thinsp;000 NIfTI image files each of resolution `32×32×32`, outputting the files to the directory at path `{nifti_directory}` run

```bash
python scripts/generate_synthetic_data.py \
    --voxels_per_axis 32 --number_of_files 10000 --output_directory {nifti_directory}
```


## Model training

A script [`train_vae_model.py`](scripts/train_vae_model.py) is included in the `scripts` directory for training variational autoencoder models on the UK Biobank FLAIR image data. Three pre-defined model configurations are given in the `example_configurations` directory as _JavaScript Object Notation_ (JSON) files &mdash; `VeryDeepVAE_32x32x32.json`, `VeryDeepVAE_64x64x64.json` and `VeryDeepVAE_128x128x128.json` &mdash; these differ only in the target resolution of the generated images (respectively `32×32×32`, `64×64×64` and `128×128×128`), the batch size used in training and the number and dimensions of the layers in the autoencoder model (see [_Layer definitions_](#layer-definitions) below), with the `64×64×64` configuration having one more layer than the `32×32×32` configuration and the `128×128×128` configuration having one more layer again than the `64×64×64` configuration.

All three example model configurations have specified to have a peak GPU memory usage of (just) less than 32GiB, so should be runnable on a GPU with 32GiB of device memory or above. To run on a GPU with less memory, either the batch size should be reduced using the `batch_size` [hyperparameter](#hyperparameters) or the latent dimensionality using the `latent_per_channels` hyperparameter - see the [_Layer definitions_ section below](#layer-definitions) for more details.

New model configurations can be specified by creating a JSON file following the structure of the included examples to define the hyperparameter values specifying the model and training configuration. See the [_Hyperparameters_ section below](#hyperparameters) for details of some of the more important properties.

### Example usages

In the below `{config_file}` should be replaced with the path to the relevant JSON file for the model configuration to train (for example `example_configurations/VeryDeepVAE_32x32x32.json`), `{nifti_directory}` with the path to the directory containing the NIfTI files to use as the trainining and validation data, and `{output_directory}` by the path to the root directory to save all model outputs to during training. In all cases it is assumed the commands are being executed in a Unix shell such as `sh` or `bash` - if using an alternative command-line interpreter such as `cmd.exe` or PowerShell on Windows the commands will not work.

To see the full set of command line arguments that can be passed to the training script run

```bash
python scripts/train_vae_model.py --help
```

#### Running on a single GPU

To run on one GPU:

```sh 
python scripts/train_vae_model.py --json_config_file {config_file} \
  --nifti_dir {nifti_directory} --output_dir {output_directory}
```
  
#### Running on multiple GPUs

To run on a single node with 8 GPU devices:

```sh
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  scripts/train_vae_model.py --json_config_file {config_file} --nifti_dir {nifti_directory} \
  --output_dir {output_directory} --CUDA_devices 0 1 2 3 4 5 6 7
```

To specify the backend and endpoint:

```sh
python -m torch.distributed.run \ 
  --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint={endpoint} \
  scripts/train_vae_model.py --json_config_file {config_file} --nifti_dir {nifti_directory} \
  --output_dir {output_directory} --CUDA_devices 0 1 2 3 4 5 6 7
```
where `{endpoint}` is the endpoint where the rendezvous backend is running in the form `host_ip:port`.

#### Running on multiple nodes

To run on two nodes, each with 8 GPU devices:

_On first node_

```sh
python -m torch.distributed.run \ 
  --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr={ip_address} --master_port={port_number} \
  scripts/train_vae_model.py --json_config_file {config_file} --nifti_dir {nifti_directory} \
  --output_dir {output_directory} --CUDA_devices 0 1 2 3 4 5 6 7 \ 
  --master_addr {ip_address} --master_port {port_number}
```
_On second node_

```sh
python -m torch.distributed.run \ 
  --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr={ip_address} --master_port={port_number} \
  scripts/train_vae_model.py --json_config_file {config_file} --nifti_dir {nifti_directory} \
  --output_dir {output_directory} --CUDA_devices 0 1 2 3 4 5 6 7 \
  --master_addr {ip_address} --master_port {port_number}
```

where `{ip_address}` is the IP address of the rank 0 node and `{port_number}` is a free 
port on the rank 0 node. 

## Model configuration

The properties specifying the model and training run configuration are specified in a JSON file. The [`model_configuration.schema.json`](model_configuration.schema.json) file in the root of the repository is a [JSON Schema](https://json-schema.org/) describing the properties which can be set in a model configuration file, the values which they can be validly set to and the default values used if properties are not explicitly set. A human-readable summary can be viewed at [`model_configuration_schema.md`](model_configuration_schema.md).

As a brief summary, some of the more important properties which you may wish to edit are

- `batch_size`:
  The number of images per minibatch for the stochastic gradient descent training algorithm. For the `128×128×128` configuration the model a batch size of 1 is needed to keep the peak GPU memory use below 32GiB. Higher batch sizes are possible at lower resolutions or on GPUs with more device memory.
- `max_niis_to_use`:
  The maximum number of NiFTI files to use in a training epoch. Use this to define a shorter epoch, for example to quickly test visualisations are being saved correctly.
- `resolution`:
  Specifies the target resolution to generate images at along each of the three image dimensions, for example `128` for a `128×128×128` resolution. Must be a positive integer power of 2.
- `visualise_training_pipeline_before_starting`:
  Set this to `true` to see a folder (`pipeline_test`, in the output folder) of augmented examples.
- `verbose`:
  Set this to `true` to get more detailed printed output during training.

### Layer definitions

The model architecture is specified by a series of properties `channels`, `channels_top_down`, `channels_hidden`, `channels_hidden_top_down`, `channels_per_latent`, `latent_feature_maps_per_resolution`, `kernel_sizes_bottom_up` and `kernel_sizes_top_down`, each of which is list of `k + 1` integers where `k` is the base-2 logarithm of the value of `resolution` - for example for the `128×128×128` configuration with `resolution` equal to 128, `k = 7`. The corresponding entries in all lists define a convolution block and after each of these we downsample by a factor of two in each spatial dimension on the way up and upsample by a factor of two in each spatial dimension on the way back down. This version of the code has not been tested when these lists have fewer than `k + 1` elements - you have been warned!

As an example the definition for the example `128×128×128` configuration is

```JSON
"channels": [20, 40, 60, 80, 100, 120, 140, 160],
"channels_top_down": [20, 40, 60, 80, 100, 120, 140, 160],
"channels_hidden": [20, 40, 60, 80, 100, 120, 140, 160],
"channels_hidden_top_down": [20, 40, 60, 80, 100, 120, 140, 160],
"channels_per_latent": [20, 20, 20, 20, 20, 20, 20, 200],
"latent_feature_maps_per_resolution": [2, 7, 6, 5, 4, 3, 2, 1],
"kernel_sizes_bottom_up": [3, 3, 3, 3, 3, 3, 2, 1],
"kernel_sizes_top_down": [3, 3, 3, 3, 3, 3, 2, 1]
```

where for example the first line specifies that, reading left to right, we have 20 output channels in the residual network block at the `128×128×128` resolution, 40 output channels in the residual network block at `64×64×64` resolution, 60 output channels in the residual network block at `32×32×32` resolution, 80  output channels in the residual network block at `16×16×16` resolution and so on.

## Authors

Robert Gray, Matt Graham, M. Jorge Cardoso, Sebastien Ourselin, Geraint Rees, Parashkev Nachev

## Funders

The Wellcome Trust, the UCLH NIHR Biomedical Research Centre

## Licence

The code is under the [GNU General Public License Version 3](LICENSE).

## References

  1. <a id='child2021very'></a> Child, R., 2021. 
  Very deep VAEs generalize autoregressive models and can outperform them on images.  
  In _Proceedings of the 9th International Conference on Learning Representations (ICLR)_.
  [(OpenReview)](https://openreview.net/forum?id=RLRXCV6DbEJ) 
  [(arXiv)](https://arxiv.org/abs/2011.10650)
  
