"""Script to train variational autoencoder model on neuroimaging data"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
import torch
from verydeepvae.orchestration import training_script_vae_new as training_script


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command line arguments."""    
    parser = argparse.ArgumentParser(
        "Train variational autoencoder model on neuroimaging data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--json_config_file", 
        type=Path,
        required=True,
        help="Path to JSON file specifying model and run hyperparameters"
    )
    parser.add_argument(
        "--nifti_flair_dir", 
        type=Path,
        required=True,
        help="Path to directory containing FLAIR image NIfTI files to train model with",
    )
    parser.add_argument(
        "--nifti_flair_pattern",
        type=str,
        default="*_flair.nii",
        help=(
            "Filename pattern for NIfTI image files to train model with. * matches "
            "everything, ? matches any single character, [seq] matches any character "
            "in seq, [!seq] matches any character not in seq"
        ),
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        required=True,
        help="Directory to save run outputs to"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0, 
        help="Rank of node when running on multiple nodes in parallel"
    )
    parser.add_argument(
        "--CUDA_devices", 
        type=str,
        nargs="+",
        default=["0"],
        help="Device indices (zero-based) for GPUs to use when training model"
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="IP address of rank 0 node when running on multiple nodes in parallel"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=1234,
        help="Free port to use on rank 0 node when running on multiple nodes in parallel"
    )
    parser.add_argument(
        "--workers_per_process",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading, set to 0 to load in main process"
    )
    parser.add_argument(
        "--threads_per_rank",
        type=int,
        default=min(8, torch.multiprocessing.cpu_count()),
        help="Number of threads to use per rank for intraop parallelism on CPU"
    )
    return parser.parse_args()


def post_process_hyperparameters(hyperparameters: Dict[str, Any], cli_args: argparse.Namespace):
    """Update loaded hyperparameter dictionary in-place using CLI argument values."""
    hyperparameters["model_name"] = cli_args.json_config_file.stem
    hyperparameters["checkpoint_folder"] = str(cli_args.output_dir / "torch_checkpoints")
    hyperparameters["tensorboard_dir"] = str(cli_args.output_dir / "tensorboard")
    hyperparameters["recon_folder"] = str(cli_args.output_dir / "reconstructions")
    for key in [
        "nifti_flair_dir", 
        "nifti_flair_pattern",
        "CUDA_devices", 
        "local_rank", 
        "master_addr", 
        "master_port", 
        "workers_per_process", 
        "threads_per_rank",
    ]:
        hyperparameters[key] = getattr(cli_args, key)
    # For hyperparameters which require specifying a boolean flag per channel or latent 
    # dimension we optionally allow using the string shorthands "all" or "none" to indicate 
    # the value for all dimensions are True or False respectively
    for key, size_as_function_of_latents_per_channel in [
        ("latents_per_channel_weight_sharing", len),
        ("latents_to_use", sum),
        ("latents_to_optimise", sum),
    ]:
        if (
            not isinstance(hyperparameters[key], list) 
            and hyperparameters[key] in {"none", "all"}
        ):
            assert "latents_per_channel" in hyperparameters, (
                "latents_per_channel must be specified in configuration file "
                f"if {key} is either 'none' or 'all'"
            )
            hyperparameters[key] = (
                [hyperparameters[key] == "all"] * 
                size_as_function_of_latents_per_channel(hyperparameters['latents_per_channel'])
            )


def main():
    cli_args = parse_command_line_arguments()
    if not cli_args.json_config_file.exists():
        raise ValueError(f"No configuration file found at {cli_args.json_config_file}")
    if not cli_args.nifti_flair_dir.exists():
        raise ValueError(f"nift_flair_dir {cli_args.nifti_flair_dir} does not exist")
    if not cli_args.nifti_flair_dir.is_dir():
        raise ValueError(
            f"nifti_flair_dir {cli_args.nifti_flair_dir} is not a directory"
        )
    if not cli_args.output_dir.exists():
        os.makedirs(cli_args.output_dir)
    with open(cli_args.json_config_file, "r") as f:
        try:
            hyperparameters = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Configuration file {cli_args.json_config_file} not valid JSON"
            ) from e
    post_process_hyperparameters(hyperparameters, cli_args)
    torch.multiprocessing.set_start_method("spawn")
    training_script.main(hyperparameters)    


if __name__ == "__main__":
    main()
