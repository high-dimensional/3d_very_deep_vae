import os
import argparse
import torch
from verydeepvae.orchestration import training_script_vae_new as training_script

current_dir = os.path.dirname(os.path.realpath('__file__'))
model_name = os.path.split(current_dir)[-1]

hyper_params = {'total_epochs': 100000, 'batch_size': 10, 'l2_reg_coeff': 1e-4,
                'learning_rate': 1e-3, 'train_frac': 0.95, 'half_precision': False, 'print_model': False,
                'current_dir': current_dir, 'model_name': model_name}

hyper_params['use_tanh_output'] = True
hyper_params['new_model'] = True
hyper_params['use_abs_not_square'] = False
hyper_params['plot_gradient_norms'] = True

hyper_params['plot_recons_period'] = 1
hyper_params['subjects_to_plot'] = 10
hyper_params['validation_period'] = 1
hyper_params['save_period'] = 1
hyper_params['base_recons_on_train_loader'] = False
hyper_params['resume_from_checkpoint'] = False
hyper_params['restore_optimiser'] = True
hyper_params['keep_every_checkpoint'] = True
hyper_params['warmup_iterations'] = 50
hyper_params['gradient_clipping_value'] = 100
hyper_params['gradient_skipping_value'] = 1000000000000

hyper_params['likelihood'] = 'Gaussian'
hyper_params['predict_x_var'] = True

hyper_params['predict_x_var_with_sigmoid'] = True
hyper_params['variance_hidden_clamp_bounds'] = [0.001, 1]
hyper_params['variance_output_clamp_bounds'] = [0.01, 1]
hyper_params['use_precision_reweighting'] = False
hyper_params['separate_hidden_loc_scale_convs'] = False
hyper_params['separate_output_loc_scale_convs'] = False
hyper_params['verbose'] = True

# hyper_params['latents_per_channel'] = [2, 7, 6, 5, 4, 3, 2, 1]
# hyper_params['channels_per_latent'] = [20, 20, 20, 20, 20, 20, 20, 200]
# hyper_params['channels'] = [20, 40, 60, 80, 100, 120, 140, 160]
# hyper_params['kernel_sizes_bottom_up'] = [3, 3, 3, 3, 3, 3, 2, 1]
# hyper_params['kernel_sizes_top_down'] = [3, 3, 3, 3, 3, 3, 2, 1]
hyper_params['latents_per_channel'] = [2, 6, 5, 4, 3, 2, 1]
hyper_params['channels_per_latent'] = [20, 20, 20, 20, 20, 20, 200]
hyper_params['channels'] = [20, 40, 60, 80, 100, 120, 140]
hyper_params['kernel_sizes_bottom_up'] = [3, 3, 3, 3, 3, 2, 1]
hyper_params['kernel_sizes_top_down'] = [3, 3, 3, 3, 3, 2, 1]
hyper_params['channels_hidden'] = hyper_params['channels']
hyper_params['channels_top_down'] = hyper_params['channels']
hyper_params['channels_hidden_top_down'] = hyper_params['channels_hidden']
hyper_params['latents_per_channel_weight_sharing'] = [False for a in hyper_params['latents_per_channel']]
hyper_params['convolutional_downsampling'] = False
hyper_params['bottleneck_resnet_encoder'] = True
hyper_params['only_use_one_conv_block_at_top'] = False  # So the 'dense layers' aren't too numerous
hyper_params['normalise_weight_by_depth'] = True
hyper_params['zero_biases'] = True
hyper_params['use_rezero'] = False
hyper_params['veto_batch_norm'] = True
hyper_params['latents_to_use'] = [True for _ in range(sum(hyper_params['latents_per_channel']))]
hyper_params['latents_to_optimise'] = [True for _ in range(sum(hyper_params['latents_per_channel']))]

hyper_params['sequence_type'] = 'flair'

hyper_params['veto_transformations'] = False
hyper_params['apply_augmentations_to_validation_set'] = False
hyper_params['visualise_training_pipeline_before_starting'] = False
# hyper_params['nifti_flair_dir'] = '/media/robert/Data2/Biobank_FLAIRs_for_VAE/'
hyper_params['nifti_flair_dir'] = '/media/robert/Data2/T1_FLAIR_seg_triples/'
hyper_params['max_niis_to_use'] = 200
hyper_params['discard_abnormally_small_niftis'] = True
hyper_params['use_nii_data'] = True
hyper_params['nii_target_shape'] = [64, 64, 64]
hyper_params['nifti_standardise'] = True
hyper_params['shuffle_niftis'] = False
hyper_params['save_recons_to_mat'] = False
hyper_params['checkpoint_folder'] = '/home/robert/temp/Torch_Checkpoints/'
hyper_params['tensorboard_dir'] = '/home/robert/temp/Torch_TensorBoard/'
# hyper_params['checkpoint_folder'] = '/local_dir/Torch_Checkpoints/'
# hyper_params['tensorboard_dir'] = '/local_dir/Torch_TensorBoard/'

hyper_params['CUDA_devices'] = [str(x) for x in range(2)]
#hyper_params['CUDA_devices'] = ['4', '5', '6', '7']
# hyper_params['CUDA_devices'] = ['0']

hyper_params['current_dir'] = current_dir
hyper_params['model_name'] = model_name

# Distributed data parallel options
hyper_params['use_DDP'] = True
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--CUDA_devices", type=str)
parser.add_argument("--nifti_flair_dir", type=str)
args = parser.parse_args()
hyper_params['args'] = args
hyper_params['master_addr'] = 'localhost'
hyper_params['master_port'] = 12345
hyper_params['workers_per_process'] = 5

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    training_script.main(hyper_params)