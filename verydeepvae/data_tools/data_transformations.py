from monai import transforms as monai_trans
from ..misc import misc
from .torch_wrapper import TorchIOWrapper
from torchio import transforms as torchio_trans
from .random_anisotropiser_augmentation import Anisotropiser
from .clamp_by_percentile_augmentation import ClampByPercentile as ClampByPercentile
from .crop_nii_by_given_amount import CropNIIByGivenAmount as CropNIIByGivenAmount
from .random_registered_3d_haircut import ThreeDHaircut


def create_data_transformations(hyper_params, device, keys=None, clamp_percentiles=None):
    """
    Create the input pipeline, with optional data augmentation
    """

    if 'clamp_percentiles' in hyper_params:
        clamp_percentiles = hyper_params['clamp_percentiles']
    else:
        if clamp_percentiles is None:
            clamp_percentiles = [0.05, 99.95]

    misc.print_0(hyper_params, "Clamping input outside percentiles " + str(clamp_percentiles[0]) + " and " +
                                                                           str(clamp_percentiles[1]))

    if keys is None:
        keys = ["full_brain"]

    resize_block = [monai_trans.Resized(keys=keys, spatial_size=tuple(hyper_params['nii_target_shape']))]

    if 'veto_transformations' in hyper_params and hyper_params['veto_transformations']:
        misc.print_0(hyper_params, "Vetoing augmentation transformations")
        train_transforms = [monai_trans.LoadImaged(keys=keys),
                            monai_trans.AddChanneld(keys=keys)]

        if 'crop_nii_at_loadtime' in hyper_params:
            print("Cropping an padding on the basis of the volume occupied by Biobank T1s at 181x217x181")
            train_transforms += [CropNIIByGivenAmount(keys=keys, crop_ranges=hyper_params['crop_nii_at_loadtime']),
                                 monai_trans.SpatialPadd(keys=keys, spatial_size=[193, 193, 193])]

        train_transforms += resize_block

        if misc.key_is_true(hyper_params, 'use_tanh_output') or misc.key_is_true(hyper_params, 'use_sigmoid_output'):
            train_transforms += [ClampByPercentile(keys=keys,
                                                   lower=clamp_percentiles[0],
                                                   upper=clamp_percentiles[1],
                                                   allow_missing_keys=True)]

        train_transforms += [monai_trans.NormalizeIntensityd(keys=keys)]

        if misc.key_is_true(hyper_params, 'use_tanh_output'):
            train_transforms += [monai_trans.ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0)]
        elif misc.key_is_true(hyper_params, 'use_sigmoid_output'):
            train_transforms += [monai_trans.ScaleIntensityd(keys=keys, minv=0, maxv=1.0)]

        train_transforms = monai_trans.Compose(train_transforms)
        val_transforms = train_transforms
    else:

        if 'min_small_crop_size' not in hyper_params:
            # This is a bit crude...
            hyper_params['min_small_crop_size'] = [int(0.95 * x) for x in hyper_params['nii_target_shape']]
            hyper_params['rot_angle_in_rads'] = 2 * 3.14159 / 360 * (5)
            hyper_params['shear_angle_in_rads'] = 2 * 3.14159 / 360 * (5)
            hyper_params['translate_range'] = 10
            hyper_params['scale_range'] = 0.1
            hyper_params['three_d_deform_sigmas'] = (1, 3)
            hyper_params['three_d_deform_magnitudes'] = (3, 5)
            hyper_params['histogram_shift_control_points'] = (10, 15)
            hyper_params['anisotroper_ranges'] = [0.8, 0.95]
            hyper_params['haircut_ranges'] = [3, 3, 3]
            hyper_params['prob_affine'] = 0.1
            hyper_params['prob_torchvision_simple'] = 0.1
            hyper_params['prob_three_d_elastic'] = 0.1
            hyper_params['prob_torchvision_histogram'] = 0.1
            hyper_params['prob_torchvision_complex'] = 0.1
            hyper_params['prob_spiking'] = 0.1
            hyper_params['prob_anisotroper'] = 0.1
            hyper_params['prob_haircut'] = 0.1

        min_small_crop_size = hyper_params['min_small_crop_size']
        rot_angle_in_rads = hyper_params['rot_angle_in_rads']
        shear_angle_in_rads = hyper_params['shear_angle_in_rads']
        translate_range = hyper_params['translate_range']
        scale_range = hyper_params['scale_range']
        three_d_deform_sigmas = hyper_params['three_d_deform_sigmas']
        three_d_deform_magnitudes = hyper_params['three_d_deform_magnitudes']
        histogram_shift_control_points = hyper_params['histogram_shift_control_points']
        anisotroper_ranges = hyper_params['anisotroper_ranges']
        haircut_ranges = hyper_params['haircut_ranges']
        prob_affine = hyper_params['prob_affine']
        prob_torchvision_simple = hyper_params['prob_torchvision_simple']
        prob_three_d_elastic = hyper_params['prob_three_d_elastic']
        prob_torchvision_histogram = hyper_params['prob_torchvision_histogram']
        prob_torchvision_complex = hyper_params['prob_torchvision_complex']
        prob_spiking = hyper_params['prob_spiking']
        prob_anisotroper = hyper_params['prob_anisotroper']
        prob_haircut = hyper_params['prob_haircut']

        train_transforms = [monai_trans.LoadImaged(keys=keys), monai_trans.AddChanneld(keys=keys)]

        if 'crop_nii_at_loadtime' in hyper_params:
            print("Cropping an padding on the basis of the voplume ocupied by Biobank T1s at 181x217x181")
            train_transforms += [CropNIIByGivenAmount(keys=keys, crop_ranges=hyper_params['crop_nii_at_loadtime']),
                                 monai_trans.SpatialPadd(keys=keys, spatial_size=[193, 193, 193])]

        train_transforms += resize_block

        train_transforms += [Anisotropiser(keys=keys, scale_range=anisotroper_ranges, prob=prob_anisotroper),
                             ThreeDHaircut(keys=keys, range=haircut_ranges, prob=prob_haircut)]

        if misc.key_is_true(hyper_params, 'use_tanh_output') or misc.key_is_true(hyper_params, 'use_sigmoid_output'):
            train_transforms += [ClampByPercentile(keys=keys,
                                                   lower=clamp_percentiles[0], upper=clamp_percentiles[1])]
        train_transforms += [monai_trans.NormalizeIntensityd(keys=keys)]
        if misc.key_is_true(hyper_params, 'use_tanh_output'):
            train_transforms += [monai_trans.ScaleIntensityd(keys, minv=-1.0, maxv=1.0)]
        elif misc.key_is_true(hyper_params, 'use_sigmoid_output'):
            train_transforms += [monai_trans.ScaleIntensityd(keys, minv=0, maxv=1.0)]

        train_transforms += [monai_trans.RandHistogramShiftd(keys=keys,
                                                             num_control_points=histogram_shift_control_points,
                                                             prob=prob_torchvision_histogram),
                             monai_trans.RandScaleIntensityd(keys=keys, factors=0.05,
                                                             prob=prob_torchvision_simple),
                             # monai_trans.RandGibbsNoised(keys=keys, prob=prob_torchvision_simple,
                             #                             alpha=(0.0, 1.0), as_tensor_output=False,
                             #                             allow_missing_keys=False),
                             # monai_trans.RandKSpaceSpikeNoised(keys, global_prob=1.0, prob=prob_torchvision_complex,
                             #                                   img_intensity_range=None,
                             #                                   label_intensity_range=None, channel_wise=True,
                             #                                   common_sampling=False, common_seed=42,
                             #                                   as_tensor_output=True, allow_missing_keys=True)
                             ]
        train_transforms += [
            monai_trans.ToTensord(keys=keys),
            TorchIOWrapper(keys=keys, p=prob_torchvision_simple,
                           trans=torchio_trans.RandomBlur(std=(0.1, 0.5))),
            TorchIOWrapper(keys=keys, p=prob_torchvision_simple,
                           trans=torchio_trans.RandomNoise(mean=0, std=(0.01, 0.3))),
            TorchIOWrapper(keys=keys, p=prob_torchvision_complex,
                           trans=torchio_trans.RandomGhosting(num_ghosts=(4, 10))),
            TorchIOWrapper(keys=keys, p=prob_torchvision_complex,
                           trans=torchio_trans.RandomBiasField(coefficients=0.5)),
            TorchIOWrapper(keys=keys, p=prob_torchvision_complex,
                           trans=torchio_trans.RandomMotion(num_transforms=1)),
            TorchIOWrapper(keys=keys, p=prob_spiking, trans=torchio_trans.RandomSpike(intensity=(1, 2), p=1)),
            monai_trans.ToTensord(keys=keys)
        ]

        train_transforms += [monai_trans.RandAffined(keys=keys,
                                                     prob=prob_affine,
                                                     rotate_range=rot_angle_in_rads,
                                                     shear_range=shear_angle_in_rads,
                                                     translate_range=translate_range,
                                                     scale_range=scale_range,
                                                     spatial_size=None,
                                                     padding_mode="border",
                                                     device=device if misc.key_is_true(hyper_params, 'affine_and_elastic_on_gpu') else None,
                                                     as_tensor_output=False),
                             monai_trans.RandSpatialCropd(keys=keys,
                                                          roi_size=min_small_crop_size, random_center=True,
                                                          random_size=False),
                             monai_trans.Rand3DElasticd(keys=keys,
                                                        sigma_range=three_d_deform_sigmas,
                                                        magnitude_range=three_d_deform_magnitudes,
                                                        prob=prob_three_d_elastic,
                                                        rotate_range=None,
                                                        shear_range=None,
                                                        translate_range=None,
                                                        scale_range=None,
                                                        spatial_size=None,
                                                        padding_mode="border",
                                                        device=device if misc.key_is_true(hyper_params, 'affine_and_elastic_on_gpu') else None,
                                                        as_tensor_output=False)
                             ]

        train_transforms += [
            monai_trans.Resized(keys=keys, spatial_size=tuple(hyper_params['nii_target_shape'])),
            monai_trans.NormalizeIntensityd(keys=keys)
        ]

        if misc.key_is_true(hyper_params, 'use_tanh_output'):
            train_transforms += [monai_trans.ScaleIntensityd(keys, minv=-1.0, maxv=1.0)]
        elif misc.key_is_true(hyper_params, 'use_sigmoid_output'):
            train_transforms += [monai_trans.ScaleIntensityd(keys, minv=0, maxv=1.0)]

        train_transforms = monai_trans.Compose(train_transforms)

        if misc.key_is_true(hyper_params, 'apply_augmentations_to_validation_set'):
            misc.print_0(hyper_params, "Applying augmentation transformations to validation set")
            val_transforms = train_transforms  # For visualisation
        else:
            val_transforms = [monai_trans.LoadImaged(keys=keys), monai_trans.AddChanneld(keys=keys)]

            if 'crop_nii_at_loadtime' in hyper_params:
                print("Cropping an padding on the basis of the voplume ocupied by Biobank T1s at 181x217x181")
                val_transforms += [
                    CropNIIByGivenAmount(keys=keys, crop_ranges=hyper_params['crop_nii_at_loadtime']),
                    monai_trans.SpatialPadd(keys=keys, spatial_size=[193, 193, 193])]

            val_transforms += resize_block

            if misc.key_is_true(hyper_params, 'use_tanh_output') or \
                    misc.key_is_true(hyper_params, 'use_sigmoid_output'):
                val_transforms += [ClampByPercentile(keys=keys,
                                                     lower=clamp_percentiles[0],
                                                     upper=clamp_percentiles[1])]
            val_transforms += [monai_trans.NormalizeIntensityd(keys=keys)]
            if misc.key_is_true(hyper_params, 'use_tanh_output'):
                val_transforms += [monai_trans.ScaleIntensityd(keys, minv=-1.0, maxv=1.0)]
            elif misc.key_is_true(hyper_params, 'use_sigmoid_output'):
                val_transforms += [monai_trans.ScaleIntensityd(keys, minv=0, maxv=1.0)]

            val_transforms = monai_trans.Compose(val_transforms)

    return val_transforms, train_transforms
