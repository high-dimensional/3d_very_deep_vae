from monai import transforms as monai_trans
import misc
from clamp_by_percentile_augmentation import ClampByPercentile as ClampByPercentile
# from NotNeeded.SimulateMissingnessAugmentation_v2 import SimulateMissingnessAugmentation
# from NotNeeded.SimulateJSONMissingnessAugmentation import SimulateJSONMissingnessAugmentation
# from NotNeeded.LoadJSONd import LoadJSONd


def create_data_transformations_2d(hyper_params, device, keys=None, clamp_percentiles=None, keys_json=None, keys_imaging=None):
    """
    ClampByPercentile should have upper limit as 99.95
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

    # PNG-specific data transforms
    train_transforms = []
    train_transforms += [monai_trans.LoadImaged(keys=keys),
                         monai_trans.AddChanneld(keys=keys)]

    # train_transforms += [ClampByPercentile(keys=keys, lower=clamp_percentiles[0], upper=clamp_percentiles[1])]
    # train_transforms += [monai_trans.NormalizeIntensityd(keys=keys)]
    # if misc.key_is_true(hyper_params, 'use_tanh_output'):
    #     train_transforms += [monai_trans.ScaleIntensityd(keys, minv=-1.0, maxv=1.0)]
    # elif misc.key_is_true(hyper_params, 'use_sigmoid_output'):
    #     train_transforms += [monai_trans.ScaleIntensityd(keys, minv=0, maxv=1.0)]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Only applying augmentation to x0 and x1")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    train_transforms += [ClampByPercentile(keys=['x0', 'x1'], lower=clamp_percentiles[0], upper=clamp_percentiles[1])]
    train_transforms += [monai_trans.NormalizeIntensityd(keys=['x0', 'x1'])]
    if misc.key_is_true(hyper_params, 'use_tanh_output'):
        train_transforms += [monai_trans.ScaleIntensityd(['x0', 'x1'], minv=-1.0, maxv=1.0)]
    elif misc.key_is_true(hyper_params, 'use_sigmoid_output'):
        train_transforms += [monai_trans.ScaleIntensityd(['x0', 'x1'], minv=0, maxv=1.0)]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Scaling x2 to [0, 1]")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    train_transforms += [monai_trans.ScaleIntensityd(['x2'], minv=0, maxv=1.0)]

    if misc.key_is_true(hyper_params, 'simulate_missingness'):
        train_transforms += [SimulateMissingnessAugmentation(keys=keys, keys_imaging=keys_imaging)]

    if keys_json is not None:
        train_transforms += [LoadJSONd(keys=keys_json)]

    if keys_json is not None and misc.key_is_true(hyper_params, 'simulate_missingness'):
        train_transforms += [SimulateJSONMissingnessAugmentation(keys=keys_json)]

    train_transforms = monai_trans.Compose(train_transforms)
    val_transforms = train_transforms

    return val_transforms, train_transforms
