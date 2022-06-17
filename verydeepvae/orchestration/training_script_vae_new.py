import glob
import os
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import numpy as np
import csv
from ..misc import visuals
from ..misc import misc
from ..misc import environment
from ..orchestration import one_epoch
from monai.data import Dataset
from ..graphs.vdeepvae_bottom_up_graph_translator import Graph as BottomUpGraph
from ..graphs.vdeepvae_top_down_graph_translator import Graph as TopDownGraph
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import math as maths
import nibabel as nib
from ..data_tools.data_transformations import create_data_transformations
import monai


def main(hyper_params):
    """
    This script coordinates everything!

    Reproducibility...
    Take another look at this:
    https://albertcthomas.github.io/good-practices-random-number-generators/

    monai.utils.set_determinism(seed=True) results in each process applying the same
    augmentation functions, which is wrong but not fatal...

    Setting torch.manual_seed doesn't seem to make a difference
    """
    random_seed = hyper_params.get("random_seed", 666)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # torch.manual_seed(random_seed)
    monai.utils.set_determinism(seed=random_seed, additional_settings=None)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    hyper_params = environment.setup_environment(hyper_params)

    if hyper_params["local_rank"] == 0:
        writer = SummaryWriter(log_dir=hyper_params["tensorboard_dir"])
    else:
        writer = None

    if "nii_target_shape" in hyper_params:
        data_shape = hyper_params["nii_target_shape"]
    else:
        misc.print_0(
            hyper_params,
            "You must specify the resample target shape using the data_shape key!",
        )
        quit()

    if hyper_params["resume_from_checkpoint"]:
        state_dict_fullpath = os.path.join(
            hyper_params["checkpoint_folder"], "state_dictionary.pt"
        )
        misc.print_0(hyper_params, "Resuming from checkpoint: " + state_dict_fullpath)

        if not os.path.exists(state_dict_fullpath):
            misc.print_0(hyper_params, "Checkpoint not found: " + state_dict_fullpath)
            quit()

        checkpoint = torch.load(state_dict_fullpath, map_location="cpu")
        filenames_flair = checkpoint["filenames_flair"]
        misc.print_0(hyper_params, "Number of niftis: " + str(len(filenames_flair)))
        nifti_paths_flair = [
            hyper_params["nifti_dir"] / name for name in filenames_flair
        ]
        nifti_b1000_filenames = filenames_flair
        nifti_b1000_paths = nifti_paths_flair
    else:
        misc.print_0(
            hyper_params,
            f"nifti_dir found in hyper_params: {hyper_params['nifti_dir']}",
        )
        nifti_paths_flair = glob.glob(
            str(hyper_params["nifti_dir"] / hyper_params["nifti_filename_pattern"])
        )
        filenames_flair = [os.path.basename(path) for path in nifti_paths_flair]
        eids = [f.split("_")[0] for f in filenames_flair]

        if misc.key_is_true(hyper_params, "load_metadata"):
            misc.print_0(
                hyper_params,
                "Loading metadata and partitioning data into normal/abnormal",
            )
            filename = (
                hyper_params["biobank_eids_dir"]
                + "biobank_eids_with_white_matter_hyperintensities.csv"
            )
            eids_with_lesions = []
            wml_volumes = []
            with open(filename) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    eids_with_lesions.append(row[0])
                    wml_volumes.append(row[1])

            del eids_with_lesions[0]
            del wml_volumes[0]
            wml_volumes = [float(v) for v in wml_volumes]

            misc.print_0(
                hyper_params,
                f"Mean wml vol: {np.mean(wml_volumes)}; std: {np.std(wml_volumes)}",
            )
            wml_mean = np.mean(wml_volumes)

            ind_normal = wml_volumes <= wml_mean
            eids_normal = [a for a, b in zip(eids_with_lesions, ind_normal) if b]
            eids_normal = [f for f in eids if f in eids_normal]

            ind_abnormal = wml_volumes > wml_mean
            eids_abnormal = [a for a, b in zip(eids_with_lesions, ind_abnormal) if b]
            eids_abnormal = [f for f in eids if f in eids_abnormal]
            misc.print_0(
                hyper_params,
                (
                    f"Number of normals: {str(len(eids_normal))}; "
                    f"number of abnormals: {str(len(eids_abnormal))}"
                ),
            )

            filenames_flair = [
                f for f in filenames_flair if f.split("_")[0] in eids_normal
            ]
            nifti_paths_flair = [
                hyper_params["nifti_dir"] / name for name in filenames_flair
            ]

        nifti_b1000_filenames = filenames_flair
        nifti_b1000_paths = nifti_paths_flair

        if "max_niis_to_use" in hyper_params:
            misc.print_0(
                hyper_params,
                "Restricting to only "
                + str(hyper_params["max_niis_to_use"])
                + " niftis",
            )
            filenames_flair = filenames_flair[0 : hyper_params["max_niis_to_use"]]
            # filenames_seg = filenames_seg[0:hyper_params['max_niis_to_use']]
            nifti_b1000_filenames = nifti_b1000_filenames[
                0 : hyper_params["max_niis_to_use"]
            ]
            nifti_paths_flair = nifti_paths_flair[0 : hyper_params["max_niis_to_use"]]
            # nifti_paths_seg = nifti_paths_seg[0:hyper_params['max_niis_to_use']]
            nifti_b1000_paths = nifti_b1000_paths[0 : hyper_params["max_niis_to_use"]]

            misc.print_0(hyper_params, "B_1000s: " + str(len(nifti_b1000_filenames)))

    training_set_size = np.floor(
        len(nifti_b1000_paths) * hyper_params["train_frac"]
    ).astype(np.int32)
    validation_set_size = len(nifti_b1000_paths) - training_set_size
    misc.print_0(hyper_params, "Training niftis: " + str(training_set_size))
    misc.print_0(hyper_params, "Validation niftis: " + str(validation_set_size))

    train_files = [
        {"full_brain": x} for x in zip(nifti_paths_flair[0:training_set_size])
    ]
    val_files = [{"full_brain": x} for x in zip(nifti_paths_flair[training_set_size::])]

    val_transforms, train_transforms = create_data_transformations(
        hyper_params, hyper_params["device"]
    )

    dataset_train = Dataset(data=train_files, transform=train_transforms)
    dataset_val = Dataset(data=val_files, transform=val_transforms)

    pin_memory = True
    cardinality_train = len(dataset_train)
    cardinality_val = len(dataset_val)
    hyper_params["cardinality_train"] = cardinality_train
    hyper_params["cardinality_val"] = cardinality_val

    is_3d = True
    is_colour = False

    batch_count_train = np.ceil(cardinality_train / hyper_params["batch_size"]).astype(
        np.int32
    )
    batch_count_val = np.ceil(cardinality_val / hyper_params["batch_size"]).astype(
        np.int32
    )

    misc.print_0(hyper_params, "Training set size: " + str(cardinality_train))
    misc.print_0(hyper_params, "Validation set size: " + str(cardinality_val))
    misc.print_0(hyper_params, "Training batches per epoch: " + str(batch_count_train))
    misc.print_0(hyper_params, "Validation batches per epoch: " + str(batch_count_val))

    hyper_params["data_shape"] = data_shape
    hyper_params["data_is_3d"] = is_3d
    hyper_params["data_is_colour"] = is_colour

    if (
        "visualise_training_pipeline_before_starting" in hyper_params
        and hyper_params["visualise_training_pipeline_before_starting"]
    ):
        misc.print_0(hyper_params, "Plotting pipeline before training")

        sampler_train = DistributedSampler(
            dataset_train,
            num_replicas=hyper_params["global_world_size"],
            rank=hyper_params["global_rank"],
            shuffle=True,
            drop_last=True,
        )
        loader_train = DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=hyper_params["batch_size"],
            drop_last=True,
            num_workers=hyper_params["workers_per_process"],
            pin_memory=pin_memory,
        )
        sampler_train.set_epoch(0)

        check_data = next(iter(loader_train))
        paths = check_data["full_brain_meta_dict"]["filename_or_obj"]
        names = [f.split("/")[-1] for f in paths]
        keys = ["full_brain"]
        to_plot = [check_data[k] for k in keys]
        titles = keys
        current_dir = os.path.join(hyper_params["recon_folder"], "pipeline_test")
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        num_to_plot = np.min(
            [hyper_params["subjects_to_plot"], hyper_params["batch_size"]]
        )

        for k in range(num_to_plot):
            nib.save(
                nib.Nifti1Image(np.squeeze(to_plot[0][k].cpu().numpy()), np.eye(4)),
                os.path.join(
                    current_dir,
                    names[k]
                    + "_full_brain"
                    + str(hyper_params["local_rank"])
                    + ".nii.gz",
                ),
            )

        visuals.plot_3d_recons_v2(
            to_plot,
            titles,
            None,
            current_dir,
            subjects_to_show=num_to_plot,
            hyper_params=hyper_params,
            prefix=str(hyper_params["local_rank"]) + "_",
        )

    dataset = [
        cardinality_train,
        dataset_train,
        cardinality_val,
        dataset_val,
        data_shape,
        is_3d,
        is_colour,
    ]

    bottom_up_graph_1 = BottomUpGraph(
        hyper_params=hyper_params, device=hyper_params["device"], input_channels=1
    )
    top_down_graph = TopDownGraph(
        hyper_params=hyper_params, device=hyper_params["device"]
    )

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=hyper_params["global_world_size"],
        rank=hyper_params["global_rank"],
        shuffle=True,
        drop_last=True,
    )
    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=hyper_params["global_world_size"],
        rank=hyper_params["global_rank"],
        shuffle=False,
        drop_last=True,
    )

    loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=hyper_params["batch_size"],
        drop_last=True,
        num_workers=hyper_params["workers_per_process"],
        pin_memory=pin_memory,
    )
    loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=hyper_params["batch_size"],
        drop_last=False,
        num_workers=hyper_params["workers_per_process"],
        pin_memory=pin_memory,
    )

    bottom_up_graph_1.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        bottom_up_graph_1.model
    )
    top_down_graph.latents = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        top_down_graph.latents
    )
    top_down_graph.x_mu = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        top_down_graph.x_mu
    )
    if (
        hyper_params["separate_output_loc_scale_convs"]
        and hyper_params["predict_x_var"]
    ):
        top_down_graph.x_var = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            top_down_graph.x_var
        )

    bottom_up_graph_1.model = torch.nn.parallel.DistributedDataParallel(
        bottom_up_graph_1.model,
        device_ids=hyper_params["device_ids"],
        output_device=hyper_params["device"],
    )
    top_down_graph.latents = torch.nn.parallel.DistributedDataParallel(
        top_down_graph.latents,
        device_ids=hyper_params["device_ids"],
        output_device=hyper_params["device"],
    )
    top_down_graph.x_mu = torch.nn.parallel.DistributedDataParallel(
        top_down_graph.x_mu,
        device_ids=hyper_params["device_ids"],
        output_device=hyper_params["device"],
    )
    if (
        hyper_params["separate_output_loc_scale_convs"]
        and hyper_params["predict_x_var"]
    ):
        top_down_graph.x_var = torch.nn.parallel.DistributedDataParallel(
            top_down_graph.x_var,
            device_ids=hyper_params["device_ids"],
            output_device=hyper_params["device"],
        )

    params = []

    if not (
        "optimise_encoder" in hyper_params and not hyper_params["optimise_encoder"]
    ):
        misc.print_0(hyper_params, "Optimising encoder")
        params += list(bottom_up_graph_1.model.parameters())

    if not ("optimise_xmu" in hyper_params and not hyper_params["optimise_xmu"]):
        params += list(top_down_graph.x_mu.parameters())

    if (
        hyper_params["separate_output_loc_scale_convs"]
        and hyper_params["predict_x_var"]
    ):
        if not ("optimise_xvar" in hyper_params and not hyper_params["optimise_xvar"]):
            params += list(top_down_graph.x_var.parameters())

    if misc.key_is_true(hyper_params, "optimise_only_prior"):
        misc.print_0(hyper_params, "Optimising only the prior in the decoder")
        params_sans_prior_predictors = []
        params_prior_predictors = []
        for name, param in top_down_graph.latents.named_parameters():
            if "convs_p" in name:
                params_prior_predictors.append(param)
            else:
                params_sans_prior_predictors.append(param)
        params += params_prior_predictors
        misc.print_0(
            hyper_params,
            "Parameters in prior being optimised: "
            + str(misc.count_parameters(params_prior_predictors)),
        )
    else:
        if not (
            "optimise_prior" in hyper_params and not hyper_params["optimise_prior"]
        ):
            misc.print_0(hyper_params, "Optimising the prior")
            params += list(top_down_graph.latents.parameters())
        else:
            misc.print_0(hyper_params, "Not optimising the prior")
            params_sans_prior_predictors = []
            params_prior_predictors = []
            for name, param in top_down_graph.latents.named_parameters():
                if "convs_p" in name:
                    params_prior_predictors.append(param)
                else:
                    params_sans_prior_predictors.append(param)
            params += params_sans_prior_predictors
            misc.print_0(
                hyper_params,
                "Ommitted parameters in prior: "
                + str(misc.count_parameters(params_prior_predictors)),
            )

    misc.print_0(
        hyper_params,
        "Parameters in bottom-up graph 1: "
        + str(
            misc.count_unique_parameters(
                list(bottom_up_graph_1.model.named_parameters())
            )
        ),
    )
    misc.print_0(
        hyper_params,
        "Parameters in top-down graph: "
        + str(
            misc.count_unique_parameters(
                list(top_down_graph.latents.named_parameters())
            )
        ),
    )
    misc.print_0(
        hyper_params,
        "Parameters in x_mu graph: "
        + str(
            misc.count_unique_parameters(list(top_down_graph.x_mu.named_parameters()))
        ),
    )
    if (
        hyper_params["separate_output_loc_scale_convs"]
        and hyper_params["predict_x_var"]
    ):
        misc.print_0(
            hyper_params,
            "Parameters in x_var graph: "
            + str(
                misc.count_unique_parameters(
                    list(top_down_graph.x_var.named_parameters())
                )
            ),
        )
    misc.print_0(
        hyper_params,
        "Total number of trainable parameters: " + str(misc.count_parameters(params)),
    )

    # optimizer = optim.Adam(list(params), lr=hyper_params['learning_rate'])
    optimizer = optim.Adamax(list(params), lr=hyper_params["learning_rate"])

    scaler = amp.GradScaler(enabled=hyper_params["half_precision"])
    if hyper_params["half_precision"]:
        misc.print_0(hyper_params, "Using AMP-based mixed precision")

    if hyper_params["resume_from_checkpoint"]:
        misc.print_0(hyper_params, "Resuming from checkpoint")
        state_dict_fullpath = os.path.join(
            hyper_params["checkpoint_folder"], "state_dictionary.pt"
        )
        checkpoint = torch.load(state_dict_fullpath, map_location="cpu")

        bottom_up_graph_1.model.load_state_dict(
            checkpoint["bottom_up_graph_1_state_dict"]
        )
        top_down_graph.latents.load_state_dict(
            checkpoint["top_down_generative_graph_state_dict"], strict=False
        )
        top_down_graph.x_mu.load_state_dict(
            checkpoint["top_down_x_mu_graph_state_dict"]
        )
        if (
            hyper_params["separate_output_loc_scale_convs"]
            and hyper_params["predict_x_var"]
        ):
            top_down_graph.x_var.load_state_dict(
                checkpoint["top_down_x_var_graph_state_dict"]
            )

        if hyper_params["restore_optimiser"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hyper_params["half_precision"]:
                scaler.load_state_dict(checkpoint["scaler"])
            starting_epoch = checkpoint["epoch"] + 1

            loss_history_train_kl = checkpoint["loss_history_train_kl"]
            loss_history_train_kl_all = checkpoint["loss_history_train_kl_all"]
            loss_history_train_mse = checkpoint["loss_history_train_mse"]
            loss_history_train_loss = checkpoint["loss_history_train_loss"]
            loss_history_train_nll_bits_per_dim = checkpoint[
                "loss_history_train_nll_bits_per_dim"
            ]
            loss_history_val_kl = checkpoint["loss_history_val_kl"]
            loss_history_val_kl_all = checkpoint["loss_history_val_kl_all"]
            loss_history_val_mse = checkpoint["loss_history_val_mse"]
            loss_history_val_loss = checkpoint["loss_history_val_loss"]
            loss_history_val_nll_bits_per_dim = checkpoint[
                "loss_history_val_nll_bits_per_dim"
            ]

            misc.print_0(
                hyper_params,
                "Previous (approximate) train loss: "
                + str(loss_history_train_loss[-1][1]),
            )
            misc.print_0(
                hyper_params,
                "Previous (approximate) validation loss: "
                + str(loss_history_val_loss[-1][1]),
            )

        else:
            misc.print_0(hyper_params, "Resetting optimiser")
            starting_epoch = 1
            loss_history_train_kl = []
            loss_history_train_kl_all = {}
            loss_history_train_mse = []
            loss_history_train_loss = []
            loss_history_train_nll_bits_per_dim = []
            loss_history_val_kl = []
            loss_history_val_kl_all = {}
            loss_history_val_mse = []
            loss_history_val_loss = []
            loss_history_val_nll_bits_per_dim = []

    else:
        starting_epoch = 1
        loss_history_train_kl = []
        loss_history_train_kl_all = {}
        loss_history_train_mse = []
        loss_history_train_loss = []
        loss_history_train_nll_bits_per_dim = []
        loss_history_val_kl = []
        loss_history_val_kl_all = {}
        loss_history_val_mse = []
        loss_history_val_loss = []
        loss_history_val_nll_bits_per_dim = []

    for epoch in range(starting_epoch, hyper_params["total_epochs"] + 1):
        # Reduce risk of memory leaks after repeated computation of recons, samples etc
        torch.cuda.empty_cache()

        sampler_train.set_epoch(epoch)  # Shuffle each epoch

        epoch_dict = {
            "bottom_up_graph_1": bottom_up_graph_1,
            "top_down_graph": top_down_graph,
            "hyper_params": hyper_params,
            "optimizer": optimizer,
            "device_ids": hyper_params["device_ids"],
            "validation_mask": None,
            "params_sans_prior_predictors": None,
            "loader": loader_train,
            "scaler": scaler,
            "params": params,
            "epoch": epoch,
            "writer": writer,
            "training": True,
            "progress_bar_text": "Optimising",
            "summary_text_prefix": "Approx loss",
            "writer_prefix": "Training",
            "device": hyper_params["device"],
        }
        output_dict = one_epoch.go(epoch_dict)

        loss_tally_train_kl = output_dict["kl_tally"]
        loss_tally_train_kl_all = output_dict["kl_all_tallies"]
        loss_tally_train_mse = output_dict["mse_tally"]
        loss_tally_train_loss = output_dict["loss_tally"]
        loss_tally_train_nll_per_dim = output_dict["nll_tally_in_bits_per_dim"]

        loss_history_train_kl.append([epoch, loss_tally_train_kl])

        for key in loss_tally_train_kl_all:
            if key in loss_history_train_kl_all:
                loss_history_train_kl_all[key].append(
                    [epoch, loss_tally_train_kl_all[key]]
                )
            else:
                loss_history_train_kl_all[key] = [[epoch, loss_tally_train_kl_all[key]]]

        loss_history_train_mse.append([epoch, loss_tally_train_mse])
        loss_history_train_loss.append([epoch, loss_tally_train_loss])
        loss_history_train_nll_bits_per_dim.append(
            [epoch, loss_tally_train_nll_per_dim]
        )

        if (
            not hyper_params["validation_period"] == 1
            and hyper_params["local_rank"] == 0
        ):
            visuals.plot_error_curves(
                data=[loss_history_train_loss],
                labels=["training error"],
                plot_title="Training error",
                recon_folder=hyper_params["recon_folder"],
                prefix="loss_train",
            )
            visuals.plot_error_curves(
                data=[loss_history_train_kl],
                labels=["training KL"],
                plot_title="Training KL",
                recon_folder=hyper_params["recon_folder"],
                prefix="kl_train",
            )
            visuals.plot_error_curves(
                data=[loss_history_train_mse],
                labels=["training MSE"],
                plot_title="Training MSE",
                recon_folder=hyper_params["recon_folder"],
                prefix="mse_train",
            )
            visuals.plot_error_curves(
                data=[loss_history_train_nll_bits_per_dim],
                labels=["training log likelihood/dim"],
                plot_title="Training log likelihood/dim",
                recon_folder=hyper_params["recon_folder"],
                prefix="nll_per_dim_train",
            )

        # Plot the gradient norms
        if (
            hyper_params["local_rank"] == 0
            and "plot_gradient_norms" in hyper_params
            and hyper_params["plot_gradient_norms"]
        ):
            grad_norms = output_dict["gradient_norms"]
            keys = list(grad_norms.keys())
            data = [grad_norms[k] for k in keys]
            visuals.plot_error_curves(
                data=data,
                labels=keys,
                plot_title="Training gradient norms",
                recon_folder=hyper_params["recon_folder"],
                prefix="grad_norms_train",
                xlabel="Iteration",
                precision=3,
            )

        del output_dict
        torch.cuda.empty_cache()

        if epoch % hyper_params["validation_period"] == 0:
            with torch.no_grad():

                validation_mask = None
                epoch_dict = {
                    "bottom_up_graph_1": bottom_up_graph_1,
                    "top_down_graph": top_down_graph,
                    "dataset": dataset,
                    "hyper_params": hyper_params,
                    "optimizer": optimizer,
                    "validation_mask": validation_mask,
                    "loader": loader_val,
                    "scaler": scaler,
                    "params": params,
                    "epoch": epoch,
                    "writer": writer,
                    "training": False,
                    "progress_bar_text": "Validating",
                    "summary_text_prefix": "Validation",
                    "writer_prefix": "Validation",
                    "device": hyper_params["device"],
                }
                output_dict = one_epoch.go(epoch_dict)

                loss_tally_val_kl = output_dict["kl_tally"]
                loss_tally_val_kl_all = output_dict["kl_all_tallies"]
                loss_tally_val_mse = output_dict["mse_tally"]
                loss_tally_val_loss = output_dict["loss_tally"]
                loss_tally_val_nll_per_dim = output_dict["nll_tally_in_bits_per_dim"]

                loss_history_val_kl.append([epoch, loss_tally_val_kl])

                for key in loss_tally_val_kl_all:
                    if key in loss_history_val_kl_all:
                        loss_history_val_kl_all[key].append(
                            [epoch, loss_tally_val_kl_all[key]]
                        )
                    else:
                        loss_history_val_kl_all[key] = [
                            [epoch, loss_tally_val_kl_all[key]]
                        ]

                loss_history_val_mse.append([epoch, loss_tally_val_mse])
                loss_history_val_loss.append([epoch, loss_tally_val_loss])
                loss_history_val_nll_bits_per_dim.append(
                    [epoch, loss_tally_val_nll_per_dim]
                )

                if hyper_params["local_rank"] == 0:
                    if len(loss_history_val_loss) == len(loss_history_train_loss):
                        visuals.plot_error_curves(
                            data=[loss_history_train_loss, loss_history_val_loss],
                            labels=["training error", "validation error"],
                            plot_title="Train & validation error",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="loss",
                        )
                        visuals.plot_error_curves(
                            data=[loss_history_train_kl, loss_history_val_kl],
                            labels=["training KL", "validation KL"],
                            plot_title="Train & validation KL",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="kl",
                        )

                        visuals.plot_error_curves(
                            data=[loss_history_train_mse, loss_history_val_mse],
                            labels=["training MSE", "validation MSE"],
                            plot_title="Train & validation MSE",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="mse",
                        )
                        visuals.plot_error_curves(
                            data=[
                                loss_history_train_nll_bits_per_dim,
                                loss_history_val_nll_bits_per_dim,
                            ],
                            labels=[
                                "training log likelihood/dim",
                                "validation log likelihood/dim",
                            ],
                            plot_title="Train & validation log likelihood/dim",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="nll_per_dim",
                        )
                    else:
                        visuals.plot_error_curves(
                            data=[loss_history_val_loss],
                            labels=["validating error"],
                            plot_title="Validation error",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="loss_val",
                        )
                        visuals.plot_error_curves(
                            data=[loss_history_val_kl],
                            labels=["validating KL"],
                            plot_title="Validation KL",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="kl_val",
                        )
                        visuals.plot_error_curves(
                            data=[loss_history_val_mse],
                            labels=["validating MSE"],
                            plot_title="Validation MSE",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="mse_val",
                        )
                        visuals.plot_error_curves(
                            data=[loss_history_val_nll_bits_per_dim],
                            labels=["validating log likelihood/dim"],
                            plot_title="Validation log likelihood/dim",
                            recon_folder=hyper_params["recon_folder"],
                            prefix="nll_per_dim_val",
                        )

                    # Plot the separate KLs
                    dimensionalities = [
                        ((2**p) ** 3) * q
                        for p, q in zip(
                            range(len(hyper_params["channels_per_latent"])),
                            hyper_params["channels_per_latent"][::-1],
                        )
                    ]
                    dims_per_latent = []
                    for k, dims in enumerate(dimensionalities):
                        num_latents = hyper_params["latents_per_channel"][-1 - k]
                        dims_per_latent += [
                            dims
                        ] * num_latents  # Times to repeat this dims

                    keys = list(loss_history_train_kl_all.keys())
                    data = [loss_history_train_kl_all[k] for k in keys]
                    visuals.plot_error_curves(
                        data=data,
                        labels=keys,
                        plot_title="Training KLs per resolution",
                        recon_folder=hyper_params["recon_folder"],
                        prefix="kl_all_train",
                        precision=5,
                    )

                    data_normed = data[:]
                    for k in range(len(data_normed)):
                        current_dims = dims_per_latent[k]
                        data_normed[k] = [
                            [a[0], a[1] / current_dims] for a in data_normed[k]
                        ]
                    visuals.plot_error_curves(
                        data=data_normed,
                        labels=keys,
                        plot_title="Training KLs per dimension, per resolution",
                        recon_folder=hyper_params["recon_folder"],
                        prefix="kl_all_train_normed",
                        precision=8,
                    )

                    keys = list(loss_history_val_kl_all.keys())
                    data = [loss_history_val_kl_all[k] for k in keys]
                    visuals.plot_error_curves(
                        data=data,
                        labels=keys,
                        plot_title="Validation KLs per resolution",
                        recon_folder=hyper_params["recon_folder"],
                        prefix="kl_all_val",
                        precision=5,
                    )
                    data_normed = data[:]
                    for k in range(len(data_normed)):
                        current_dims = dims_per_latent[k]
                        data_normed[k] = [
                            [a[0], a[1] / current_dims] for a in data_normed[k]
                        ]
                    visuals.plot_error_curves(
                        data=data_normed,
                        labels=keys,
                        plot_title="Validation KLs per dimension, per resolution",
                        recon_folder=hyper_params["recon_folder"],
                        prefix="kl_all_val_normed",
                        precision=8,
                    )

        if hyper_params["save_period"] > 0 and epoch % hyper_params["save_period"] == 0:
            if hyper_params["local_rank"] == 0:
                misc.print_0(hyper_params, "Saving model")
                validation_mask = None
                checkpoint_dict = {
                    "hyper_params": hyper_params,
                    "epoch": epoch,
                    "validation_mask": validation_mask,
                    "bottom_up_graph_1_state_dict": bottom_up_graph_1.model.state_dict(),
                    "top_down_generative_graph_state_dict": top_down_graph.latents.state_dict(),
                    "top_down_x_mu_graph_state_dict": top_down_graph.x_mu.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_history_train_kl": loss_history_train_kl,
                    "loss_history_train_kl_all": loss_history_train_kl_all,
                    "loss_history_train_mse": loss_history_train_mse,
                    "loss_history_train_loss": loss_history_train_loss,
                    "loss_history_train_nll_bits_per_dim": loss_history_train_nll_bits_per_dim,
                    "loss_history_val_kl": loss_history_val_kl,
                    "loss_history_val_kl_all": loss_history_val_kl_all,
                    "loss_history_val_mse": loss_history_val_mse,
                    "loss_history_val_loss": loss_history_val_loss,
                    "loss_history_val_nll_bits_per_dim": loss_history_val_nll_bits_per_dim,
                }

                if "kl_multiplier" in hyper_params:
                    checkpoint_dict["kl_multiplier"] = hyper_params["kl_multiplier"]

                checkpoint_dict["filenames_flair"] = filenames_flair

                if (
                    hyper_params["separate_output_loc_scale_convs"]
                    and hyper_params["predict_x_var"]
                ):
                    checkpoint_dict[
                        "top_down_x_var_graph_state_dict"
                    ] = top_down_graph.x_var.state_dict()
                if hyper_params["half_precision"]:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if hyper_params["keep_every_checkpoint"]:
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            hyper_params["checkpoint_folder"],
                            "state_dictionary_" + str(epoch) + ".pt",
                        ),
                    )

                torch.save(
                    checkpoint_dict,
                    os.path.join(
                        hyper_params["checkpoint_folder"], "state_dictionary.pt"
                    ),
                )

        if epoch % hyper_params["plot_recons_period"] == 0:
            misc.print_0(hyper_params, "Computing training set reconstructions")
            with torch.no_grad():
                bottom_up_graph_1.model.eval()
                top_down_graph.latents.eval()
                top_down_graph.x_mu.eval()
                if (
                    hyper_params["separate_output_loc_scale_convs"]
                    and hyper_params["predict_x_var"]
                ):
                    top_down_graph.x_var.eval()

                torch.cuda.empty_cache()

                with amp.autocast(hyper_params["half_precision"]):
                    batch = next(iter(loader_val))

                    current_input = batch["full_brain"].to(hyper_params["device"])

                    if hyper_params["half_precision"]:
                        current_input = current_input.type(torch.float32)

                    current_input = current_input.cpu().detach().numpy()

                    to_plot = [current_input]
                    titles = ["input: current_input"]

                    if hyper_params["predict_x_var"]:
                        to_plot_std = [current_input]
                        titles_std = ["input: current_input"]

                    # Tell the top down block which resolutions it should sample from
                    # the posterior
                    for min in range(1, 1 + len(hyper_params["channels_per_latent"])):
                        max = len(hyper_params["channels_per_latent"])

                        if "hidden_spatial_dims" in hyper_params:
                            temp = (
                                hyper_params["nii_target_shape"][0:1]
                                + hyper_params["hidden_spatial_dims"][:]
                            )
                            res_to_sample_from_prior = temp[::-1][min:]
                        else:
                            res_to_sample_from_prior = [2**p for p in range(min, max)]

                        # In this case use dictionaries
                        current_input = batch["full_brain"].to(hyper_params["device"])

                        input_dictionary_1 = {"data": current_input}

                        data_dictionary_1 = bottom_up_graph_1.model(input_dictionary_1)

                        data_dictionary = {
                            "data": data_dictionary_1["data"],
                            "KL_list": [],
                            "res_to_sample_from_prior": res_to_sample_from_prior,
                        }
                        for key in data_dictionary_1:
                            data_dictionary["encoder1_" + key] = data_dictionary_1[key]

                        data_dictionary_latents = top_down_graph.latents(
                            data_dictionary
                        )
                        data_dictionary_x_mu = top_down_graph.x_mu(
                            data_dictionary_latents
                        )

                        x_mu, x_std, x_var, x_log_var = misc.gaussian_output(
                            data_dictionary_x_mu,
                            data_dictionary_latents,
                            top_down_graph,
                            hyper_params,
                            num_modalities=1,
                        )

                        if hyper_params["half_precision"]:
                            x_mu = x_mu.type(torch.float32)
                            if hyper_params["predict_x_var"]:
                                x_std = x_std.type(torch.float32)

                        x_mu = x_mu.cpu().detach().numpy()
                        if hyper_params["predict_x_var"]:
                            x_std = x_std.cpu().detach().numpy()

                        to_plot += [x_mu]
                        if hyper_params["predict_x_var"]:
                            to_plot_std += [x_std]

                        if min == max:
                            if is_3d:
                                titles += ["E[p(current_input | z)]. No imputation!"]
                                if hyper_params["predict_x_var"]:
                                    titles_std += [
                                        "STD[p(current_input | z)]. No imputation!"
                                    ]
                            else:
                                titles += ["E[p(current_input | z)].\nNo imputation!"]
                                if hyper_params["predict_x_var"]:
                                    titles_std += [
                                        "STD[p(current_input | z)].\nNo imputation!"
                                    ]
                        else:
                            if "hidden_spatial_dims" in hyper_params:
                                res = str(
                                    hyper_params["hidden_spatial_dims"][::-1][min - 1]
                                )
                            else:
                                res = str(2 ** (min - 1))

                            if is_3d:
                                titles += [
                                    "E[p(current_input | z)]. Imputing latents above "
                                    + res
                                    + " cubed using the prior"
                                ]
                                if hyper_params["predict_x_var"]:
                                    titles_std += [
                                        "STD[p(current_input | z)]. "
                                        + "Imputing latents above "
                                        + res
                                        + " cubed using the prior"
                                    ]
                            else:
                                titles += [
                                    "E[p(current_input | z)].\nImputing latents above "
                                    + res
                                    + "\nsquared using the prior"
                                ]
                                if hyper_params["predict_x_var"]:
                                    titles_std += [
                                        "STD[p(current_input | z)].\n"
                                        + "Imputing latents above "
                                        + res
                                        + "\nsquared using the prior"
                                    ]

                    subjects_to_plot_per_process = maths.floor(
                        hyper_params["subjects_to_plot"]
                    )
                    prefix = "prog_recons_rank" + str(hyper_params["local_rank"])
                    if hyper_params["predict_x_var"]:
                        prefix_std = "prog_stds_rank" + str(hyper_params["local_rank"])
                    if is_3d:
                        visuals.plot_3d_recons_v2(
                            to_plot,
                            titles,
                            epoch,
                            hyper_params["recon_folder"],
                            subjects_to_show=subjects_to_plot_per_process,
                            hyper_params=hyper_params,
                            prefix=prefix,
                        )
                        if hyper_params["predict_x_var"]:
                            visuals.plot_3d_recons_v2(
                                to_plot_std,
                                titles_std,
                                epoch,
                                hyper_params["recon_folder"],
                                subjects_to_show=subjects_to_plot_per_process,
                                hyper_params=hyper_params,
                                prefix=prefix_std,
                            )
                    else:
                        visuals.plot_2d(
                            to_plot,
                            titles,
                            epoch,
                            hyper_params["recon_folder"],
                            filename=prefix,
                            is_colour=False,
                            num_to_plot=subjects_to_plot_per_process,
                            norm_recons=True,
                        )
                        if hyper_params["predict_x_var"]:
                            visuals.plot_2d(
                                to_plot_std,
                                titles_std,
                                epoch,
                                hyper_params["recon_folder"],
                                filename=prefix_std,
                                is_colour=False,
                                num_to_plot=subjects_to_plot_per_process,
                                norm_recons=True,
                            )

                    # Repeat this but now just plot differences
                    subjects_to_plot_per_process = maths.floor(
                        hyper_params["subjects_to_plot"]
                    )
                    prefix = "prog_diffs_rank" + str(hyper_params["local_rank"])
                    if is_3d:
                        to_plot = [to_plot[0]] + [
                            np.abs(a - to_plot[0]) for a in to_plot[1:]
                        ]
                        visuals.plot_3d_recons_v2(
                            to_plot,
                            titles,
                            epoch,
                            hyper_params["recon_folder"],
                            subjects_to_show=subjects_to_plot_per_process,
                            hyper_params=hyper_params,
                            prefix=prefix,
                        )
                    else:
                        visuals.plot_2d(
                            to_plot,
                            titles,
                            epoch,
                            hyper_params["recon_folder"],
                            filename=prefix,
                            is_colour=False,
                            num_to_plot=subjects_to_plot_per_process,
                            norm_recons=True,
                        )

                    ####################################################################
                    ####################################################################
                    ####################################################################
                    ####################################################################

                    # Now we create samples!
                    misc.print_0(hyper_params, "\nComputing samples")
                    if not is_3d:
                        # In 2D, we compute all samples in one forward pass (per GPU)
                        times_to_sample = 1
                    elif "times_to_sample" in hyper_params:
                        times_to_sample = hyper_params["times_to_sample"]
                    else:
                        times_to_sample = 5

                    for temp in [1]:
                        temp_prefix = f"temp{temp}_"

                        for k in range(times_to_sample):
                            data_dictionary = {
                                "data": data_dictionary_1["data"],
                                "KL_list": [],
                            }
                            for key in data_dictionary:
                                if "encoder1_" in key:
                                    data_dictionary[key] = None

                            data_dictionary["sampling_noise_std_override"] = temp
                            data_dictionary_latents = top_down_graph.latents(
                                data_dictionary
                            )
                            data_dictionary_x_mu = top_down_graph.x_mu(
                                data_dictionary_latents
                            )

                            (
                                samples,
                                samples_std,
                                samples_var,
                                samples_log_var,
                            ) = misc.gaussian_output(
                                data_dictionary_x_mu,
                                data_dictionary_latents,
                                top_down_graph,
                                hyper_params,
                                num_modalities=1,
                            )

                            samples = samples.type(torch.float32)
                            samples = samples.cpu().detach().numpy()
                            if hyper_params["predict_x_var"]:
                                samples_std = samples_std.type(torch.float32)
                                samples_std = samples_std.cpu().detach().numpy()

                            if is_3d:
                                batch_shape = list(samples.shape)
                                affine = (
                                    batch["full_brain_meta_dict"]["affine"][0]
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )

                                for m in range(batch_shape[0]):
                                    current_filename = (
                                        "rank"
                                        + str(hyper_params["local_rank"])
                                        + "_example"
                                        + str(m)
                                        + ".nii"
                                    )
                                    current_vol = np.squeeze(samples[m])
                                    nim = nib.Nifti1Image(current_vol, affine)
                                    nib.save(
                                        nim,
                                        os.path.join(
                                            hyper_params["samples_folder"],
                                            current_filename,
                                        ),
                                    )

                                to_plot = [samples]
                                titles = ["sample from p(current_input | z)"]
                                to_plot += [samples_std]
                                titles += ["std of p(current_input | z)"]

                                subjects_to_plot_per_process = np.min(
                                    [
                                        maths.floor(hyper_params["subjects_to_plot"]),
                                        hyper_params["batch_size"],
                                    ]
                                )
                                prefix = temp_prefix + "samples_" + str(k) + "_"

                                current_dir = [
                                    os.path.join(
                                        hyper_params["samples_folder"],
                                        "rank"
                                        + str(hyper_params["local_rank"])
                                        + "_"
                                        + str(n),
                                    )
                                    for n in range(subjects_to_plot_per_process)
                                ]
                                for d in current_dir:
                                    if not os.path.isdir(d):
                                        os.mkdir(d)

                                visuals.plot_3d_recons_v2(
                                    to_plot,
                                    titles,
                                    epoch,
                                    current_dir,
                                    subjects_to_show=subjects_to_plot_per_process,
                                    hyper_params=hyper_params,
                                    prefix=prefix,
                                )

                            else:
                                visuals.image_grid(
                                    samples,
                                    epoch,
                                    hyper_params["recon_folder"],
                                    temp_prefix
                                    + "samples_rank"
                                    + "_"
                                    + str(hyper_params["local_rank"]),
                                    5,
                                    norm_recons=True,
                                )
                                if hyper_params["predict_x_var"]:
                                    visuals.image_grid(
                                        samples_std,
                                        epoch,
                                        hyper_params["recon_folder"],
                                        temp_prefix
                                        + "samples_stds_rank"
                                        + "_"
                                        + str(hyper_params["local_rank"]),
                                        5,
                                        norm_recons=True,
                                    )

                            del data_dictionary
                            del data_dictionary_latents
                            del data_dictionary_x_mu

                    # data_dictionary_1 is left over from the recon logic and is used
                    # each time we sample
                    del data_dictionary_1

    if writer is not None:
        writer.close()
