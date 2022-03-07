import os
import argparse
import torch
from verydeepvae.orchestration import training_script_vae_new as training_script

current_dir = os.path.dirname(os.path.realpath('__file__'))
model_name = os.path.split(current_dir)[-1]

hyper_params = {'total_epochs': 100000, 'batch_size': 2, 'l2_reg_coeff': 1e-4,
                'learning_rate': 1e-3, 'train_frac': 0.95, 'half_precision': False, 'print_model': False,
                'current_dir': current_dir, 'model_name': model_name}

hyper_params['use_tanh_output'] = True
hyper_params['new_model'] = True
hyper_params['use_abs_not_square'] = False
hyper_params['plot_gradient_norms'] = True

hyper_params['apply_mask_in_input_space'] = False
hyper_params['include_mask_in_loader'] = False

hyper_params['plot_recons_period'] = 1
hyper_params['subjects_to_plot'] = 4
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
# hyper_params['x_std_l2_penalty'] = 1e-3
hyper_params['model_dwi_given_t1'] = False
hyper_params['model_b1000_given_b10'] = False

hyper_params['predict_x_var_with_sigmoid'] = True
hyper_params['variance_hidden_clamp_bounds'] = [0.001, 1]
hyper_params['variance_output_clamp_bounds'] = [0.01, 1]
hyper_params['use_precision_reweighting'] = False
hyper_params['separate_hidden_loc_scale_convs'] = False
hyper_params['separate_output_loc_scale_convs'] = False
hyper_params['optimise_encoder'] = True
hyper_params['optimise_prior'] = True
hyper_params['optimise_xmu'] = True
hyper_params['optimise_xvar'] = True
hyper_params['optimise_only_prior'] = False
#hyper_params['kl_multiplier'] = 0.01
#hyper_params['noise_injection_multiplier'] = 0.001
hyper_params['veto_noise_injection'] = False
hyper_params['generate_mask_for_validating'] = False

hyper_params['kl_weight_auto_adjustment'] = False
hyper_params['kl_weight_adjustment_period'] = 5
hyper_params['kl_weight_epoch_patience'] = 0
hyper_params['kl_weight_max_multiplier'] = 100
hyper_params['kl_weight_min_multiplier'] = 1
hyper_params['kl_weight_multiplier_increment'] = 1
hyper_params['kl_weight_percentile_to_reweight'] = 25
hyper_params['kl_weight_verbose'] = False

hyper_params['verbose'] = True

# hyper_params['latents_per_channel'] = [2, 7, 6, 5, 4, 3, 2, 1]
# hyper_params['channels_per_latent'] = [20, 20, 20, 20, 20, 20, 20, 200]
# hyper_params['channels'] = [20, 40, 60, 80, 100, 120, 140, 160]
# hyper_params['kernel_sizes_bottom_up'] = [3, 3, 3, 3, 3, 3, 2, 1]
# hyper_params['kernel_sizes_top_down'] = [3, 3, 3, 3, 3, 3, 2, 1]
hyper_params['latents_per_channel'] = [2, 5, 4, 3, 2, 1]
hyper_params['channels_per_latent'] = [20, 20, 20, 20, 20, 200]
hyper_params['channels'] = [20, 40, 60, 80, 100, 120]
hyper_params['kernel_sizes_bottom_up'] = [3, 3, 3, 3, 2, 1]
hyper_params['kernel_sizes_top_down'] = [3, 3, 3, 3, 2, 1]
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

hyper_params['conditional_model'] = False
hyper_params['delete_hemispheres'] = False
hyper_params['leave_random_subvolumes'] = False
hyper_params['delete_random_subvolumes'] = False
hyper_params['sequence_type'] = 'flair'
hyper_params['mask_out_wmls_in_likelihood'] = False
hyper_params['load_metadata'] = False

hyper_params['veto_transformations'] = False
hyper_params['apply_augmentations_to_validation_set'] = False
hyper_params['visualise_training_pipeline_before_starting'] = True
hyper_params['nifti_flair_dir'] = '/media/robert/Data2/Biobank_FLAIRs_for_VAE/'
# hyper_params['max_niis_to_use'] = 200
hyper_params['discard_abnormally_small_niftis'] = True
hyper_params['affine_and_elastic_on_gpu'] = False
hyper_params['use_nii_data'] = True
# hyper_params['nii_target_shape'] = [128, 128, 128]
hyper_params['nii_target_shape'] = [32, 32, 32]
hyper_params['nifti_standardise'] = True
hyper_params['shuffle_niftis'] = False
hyper_params['save_recons_to_mat'] = False
hyper_params['checkpoint_folder'] = '/local_dir/Torch_Checkpoints/'
hyper_params['tensorboard_dir'] = '/local_dir/Torch_TensorBoard/'

hyper_params['RandHistogramShift_num_control_points'] = (5, 15)
hyper_params['RandHistogramShift_prob'] = 0.2
hyper_params['RandScaleIntensity_factors'] = 0.1
hyper_params['RandHistogramShift_prob'] = 0.2

hyper_params['min_small_crop_size'] = [int(0.95 * x) for x in hyper_params['nii_target_shape']]
hyper_params['rot_angle_in_rads'] = 2 * 3.14159 / 360 * ( 5 )
hyper_params['shear_angle_in_rads'] = 2 * 3.14159 / 360 * ( 5 )
hyper_params['translate_range'] = 10
hyper_params['scale_range'] = 0.1
hyper_params['three_d_deform_sigmas'] = (1, 3)
hyper_params['three_d_deform_magnitudes'] = (3, 5)
hyper_params['histogram_shift_control_points'] = (10, 15)
hyper_params['anisotroper_ranges'] = [0.8, 0.95]
hyper_params['prob_affine'] = 0.1
hyper_params['prob_torchvision_simple'] = 0.1
hyper_params['prob_three_d_elastic'] = 0.1
hyper_params['prob_torchvision_histogram'] = 0.1
hyper_params['prob_torchvision_complex'] = 0.1
hyper_params['prob_spiking'] = 0.1
hyper_params['prob_anisotroper'] = 0.1

# hyper_params['CUDA_devices'] = [str(x) for x in range(8)]
#hyper_params['CUDA_devices'] = ['4', '5', '6', '7']
hyper_params['CUDA_devices'] = ['0']

hyper_params['current_dir'] = current_dir
hyper_params['model_name'] = model_name

# Distributed data parallel options
hyper_params['use_DDP'] = True
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
hyper_params['args'] = args
hyper_params['world_size'] = len(hyper_params['CUDA_devices'])
hyper_params['master_addr'] = 'localhost'
hyper_params['master_port'] = 12345
hyper_params['workers_per_process'] = 20
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(hyper_params['CUDA_devices'])

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    training_script.main(hyper_params)

