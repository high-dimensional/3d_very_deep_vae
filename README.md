## Some notes
(Robert, r.gray@ucl.ac.uk, 5/2/2022)

All the code is in 'SharedModules'. To create a new autoencoder just create a folder alongside SharedModules and put a run_autoencoder.py file in it, just like the ones I have included (or just use the ones I have included!). The only difference between the 3 models I have included is the 'nii_target_shape' argument, and the number of layers (see 'Layer definitions' below): the 64^3 model has one more layer than the 32^3 model, and the 128^3 model has one more layer than that).

The model defined in VeryDeepVAE_128x128x128 uses between 31.9GB, so should fit on a 32GB card! Changing the 7 to a 6 in hyper_params['latents_per_channel'] should make it fit comfortably in 32GB if that becomes a problem.

- To run on one GPU (see the point below about hyper_params['CUDA_devices']):
 
python run_autoencoder.py --local_rank 0

- To run on multiple GPUs (see the point below about hyper_params['CUDA_devices']):

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 run_autoencoder.py

- To specify the backend and endpoint (see the point below about hyper_params['CUDA_devices']):

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:12345 run_autoencoder.py

- To kill zombies

kill $(ps aux | grep config.py | grep -v grep | awk '{print $2}') and kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')

## The more important hyperparameters:

- hyper_params['nifti_flair_dir']

All the Biobank niftis are loose in this folder. We have affine-aligned them to a template and skull stripped them, using SPM.

- 'batch_size'

At 128 cubed we train with a batch size of 2 or 3 on Cambridge 1, and a batch size of 1 on our V100 DGX1s. Much higher batch sizes are possible at lower resolutions!

- hyper_params['CUDA_devices']

This is a list of stringy ints, e.g. ['0'], ['0', '1'], etc. It uses zero indexing. Make sure this corresponds to what you specify on the command line - they must agree! E.g. if you intend to run on one GPU then set this to ['0'], if you intend to run on 8 GPUS then set this to [str(x) for x in range(8)], etc.

- hyper_params['max_niis_to_use']

Use this to define a shorter epoch (e.g. to quickly test visualisations are being saved correctly).

- hyper_params['nii_target_shape']

For working at different resolutions, e.g.,
hyper_params['nii_target_shape'] = [128, 128, 128]
hyper_params['nii_target_shape'] = [64, 64, 64]
hyper_params['nii_target_shape'] = [32, 32, 32]
etc

- Layer definitions: e.g.,

hyper_params['latents_per_channel'] = [2, 7, 6, 5, 4, 3, 2, 1]  # They go high res -> low res (i.e. 1 latent at 1x1x1 res, and 2 latents at input res)
hyper_params['channels_per_latent'] = [20, 20, 20, 20, 20, 20, 20, 200]  # They go high res -> low res
hyper_params['channels'] = [20, 40, 60, 80, 100, 120, 140, 160]  # They go high res -> low res
hyper_params['kernel_sizes_bottom_up'] = [3, 3, 3, 3, 3, 3, 2, 1]  # They go high res -> low res
hyper_params['kernel_sizes_top_down'] = [3, 3, 3, 3, 3, 3, 2, 1]  # They go high res -> low res

If the resolution is 2^k then these lists should all have k+1 elements: each entry defines a convolution block and after each of these we downsample by x2 in each spatial dim on the way up and upsample by x2 in each spatial dim on the way back down. This version of the code has not been tested when these lists have fewer than k+1 elements - you have been warned!

- hyper_params['visualise_training_pipeline_before_starting']

Set this to 'True' to see a folder ('pipeline_test', in the 'output' folder) of augmented examples!


## Requirements:

- (venv) robert@robert-work:~/Dropbox/PyTorch/DiRAC/3d_very_deep_vae$ pip freeze

absl-py==0.15.0
argon2-cffi==21.1.0
attrs==21.2.0
backcall==0.2.0
bleach==4.1.0
cachetools==4.2.4
certifi==2021.10.8
cffi==1.15.0
charset-normalizer==2.0.7
click==8.0.3
cycler==0.10.0
debugpy==1.5.1
decorator==5.1.0
defusedxml==0.7.1
Deprecated==1.2.13
entrypoints==0.3
google-auth==2.3.0
google-auth-oauthlib==0.4.6
grpcio==1.41.0
h5py==3.5.0
humanize==3.12.0
idna==3.3
imageio==2.9.0
importlib-resources==5.4.0
ipykernel==6.5.1
ipython==7.29.0
ipython-genutils==0.2.0
ipywidgets==7.6.5
jedi==0.18.1
Jinja2==3.0.3
jsonschema==4.2.1
jupyter-client==7.1.0
jupyter-console==6.4.0
jupyter-core==4.9.1
jupyterlab-pygments==0.1.2
jupyterlab-widgets==1.0.2
kiwisolver==1.3.2
Markdown==3.3.4
MarkupSafe==2.0.1
matplotlib==3.4.3
matplotlib-inline==0.1.3
mistune==0.8.4
monai==0.7.0
nbclient==0.5.9
nbconvert==6.3.0
nbformat==5.1.3
nest-asyncio==1.5.1
networkx==2.6.3
nibabel==3.2.1
notebook==6.4.6
numpy==1.21.3
oauthlib==3.1.1
packaging==21.0
pandocfilters==1.5.0
parso==0.8.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow==8.4.0
prometheus-client==0.12.0
prompt-toolkit==3.0.22
protobuf==3.19.0
ptyprocess==0.7.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycparser==2.21
Pygments==2.10.0
pyparsing==2.4.7
pyrsistent==0.18.0
python-dateutil==2.8.2
PyWavelets==1.1.1
pyzmq==22.3.0
qtconsole==5.2.0
QtPy==1.11.2
requests==2.26.0
requests-oauthlib==1.3.0
rsa==4.7.2
scikit-image==0.18.3
scipy==1.7.1
Send2Trash==1.8.0
SimpleITK==2.1.1
six==1.16.0
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
terminado==0.12.1
testpath==0.5.0
tifffile==2021.10.12
torch==1.10.0+cu113
torchaudio==0.10.0+cu113
torchio==0.18.57
torchvision==0.11.1+cu113
tornado==6.1
tqdm==4.62.3
traitlets==5.1.1
typing-extensions==3.10.0.2
urllib3==1.26.7
wcwidth==0.2.5
webencodings==0.5.1
Werkzeug==2.0.2
widgetsnbextension==3.5.2
wrapt==1.13.2
zipp==3.6.0

