# Some notes

All the code is in 'SharedModules'. To create a new autoencoder just create a folder alongside SharedModules and put a run_autoencoder.py file in it, just like the ones I have included (or just use the ones I have included!). The only difference between the 3 models I have included is the 'nii_target_shape' argument, and the number of layers (see 'Layer definitions' below): the 64^3 model has one more layer than the 32^3 model, and the 128^3 model has one more layer than that).

The model defined in VeryDeepVAE_128x128x128 uses between 31.9GB, so should fit on a 32GB card! Changing the 7 to a 6 in hyper_params['latents_per_channel'] should make it fit comfortably in 32GB if that becomes a problem.

- To run on one GPU (see the point below about hyper_params['CUDA_devices']):
 
python run_autoencoder.py --local_rank 0

- To run on multiple GPUs (see the point below about hyper_params['CUDA_devices']):

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 run_autoencoder.py

- To specify the backend and endpoint (see the point below about hyper_params['CUDA_devices']):

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:12345 run_autoencoder.py

- To run on two nodes, each with 8 cards

python -m torch.distributed.run --nproc_per_node=8 --master_addr=123.45.678.90 --master_port=1234 --nnodes=2 --node_rank=0 run_autoencoder.py

python -m torch.distributed.run --nproc_per_node=8 --master_addr=123.45.678.90 --master_port=1234 --nnodes=2 --node_rank=1 run_autoencoder.py

# To kill zombies

kill $(ps aux | grep config.py | grep -v grep | awk '{print $2}') and kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')

## The more important hyperparameters:

- hyper_params['nifti_flair_dir']

All the Biobank niftis are loose in this folder. We have affine-aligned them to a template and skull stripped them, using SPM.

- 'batch_size'

At 128 cubed we train with a batch size of 2 or 3 on Cambridge 1, and a batch size of 1 on our V100 DGX1s. Much higher batch sizes are possible at lower resolutions!

- CUDA_devices

Specify in hyper_params['CUDA_devices'] as a list of stringy ints, e.g. ['0'], ['0', '1'], etc, or on the command line (--CUDA_devices) as a list of ints. It uses zero indexing. Make sure this corresponds to what you specify on the command line - they must agree! E.g. if you intend to run on one GPU then set this to ['0'], if you intend to run on 8 GPUS then set this to [str(x) for x in range(8)], etc.

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


## Dependencies

To run the code a Python environment with one of Python 3.7, 3.8 or 3.9 is required. 
The Python dependencies for the code are specified in the [`requirements.txt`](requirements.txt) file.
