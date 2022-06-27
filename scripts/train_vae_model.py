"""Script to train variational autoencoder model on neuroimaging data"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
import jsonschema
import torch
from verydeepvae.orchestration.training import train_model


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        "Train variational autoencoder model on neuroimaging data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json_config_file",
        type=Path,
        required=True,
        help="Path to JSON file specifying model and run hyperparameters",
    )
    parser.add_argument(
        "--nifti_dir",
        type=Path,
        required=True,
        help="Path to directory containing NIfTI files to train & validate model with",
    )
    parser.add_argument(
        "--nifti_filename_pattern",
        type=str,
        default="*.nii",
        help=(
            "Pattern for names of NIfTI files to use to train and validate model. * "
            "matches everything, ? matches any single character, [seq] matches any  "
            "character in seq, [!seq] matches any character not in seq"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save run outputs to",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Rank of node when running on multiple nodes in parallel",
    )
    parser.add_argument(
        "--CUDA_devices",
        type=str,
        nargs="+",
        default=["0"],
        help="Device indices (zero-based) for GPUs to use when training model",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="IP address of rank 0 node when running on multiple nodes in parallel",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=1234,
        help="Port to use on rank 0 node when running on multiple nodes in parallel",
    )
    parser.add_argument(
        "--workers_per_process",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading, set to 0 to load in main "
            "process"
        ),
    )
    parser.add_argument(
        "--threads_per_rank",
        type=int,
        default=min(8, torch.multiprocessing.cpu_count()),
        help="Number of threads to use per rank for intraop parallelism on CPU",
    )
    return parser.parse_args()


def post_process_hyperparameters(
    hyperparameters: Dict[str, Any], cli_args: argparse.Namespace
):
    """Update loaded hyperparameter dictionary in-place using CLI argument values."""
    hyperparameters["model_name"] = cli_args.json_config_file.stem
    hyperparameters["checkpoint_folder"] = str(cli_args.output_dir / "checkpoints")
    hyperparameters["tensorboard_dir"] = str(cli_args.output_dir / "tensorboard")
    hyperparameters["recon_folder"] = str(cli_args.output_dir / "reconstructions")
    for key in [
        "nifti_dir",
        "nifti_filename_pattern",
        "CUDA_devices",
        "local_rank",
        "master_addr",
        "master_port",
        "workers_per_process",
        "threads_per_rank",
    ]:
        hyperparameters[key] = getattr(cli_args, key)
    # For hyperparameters which require specifying a boolean flag per channel or latent
    # dimension we optionally allow using the string shorthands "all" or "none" to
    # indicate the value for all dimensions are True or False respectively
    for key, size_as_function_of_latent_feature_maps_per_resolution in [
        ("latent_feature_maps_per_resolution_weight_sharing", len),
        ("latents_to_use", sum),
        ("latents_to_optimise", sum),
    ]:
        if not isinstance(hyperparameters[key], list) and hyperparameters[key] in {
            "none",
            "all",
        }:
            assert "latent_feature_maps_per_resolution" in hyperparameters, (
                "latent_feature_maps_per_resolution must be specified in configuration "
                f"file if {key} is either 'none' or 'all'"
            )
            hyperparameters[key] = [
                hyperparameters[key] == "all"
            ] * size_as_function_of_latent_feature_maps_per_resolution(
                hyperparameters["latent_feature_maps_per_resolution"]
            )


def load_config_schema():
    schema_path = Path(__file__).parent.parent / "model_configuration.schema.json"
    with open(schema_path, "r") as f:
        try:
            schema = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Configuration schema file {schema_path} not valid JSON"
            ) from e
    return schema


def extend_validator_with_default(validator_class):
    """Extend jsonschema validator class to set optional property values to defaults.

    Renaming of function provided in FAQ of jsonschema documentation at
    https://github.com/python-jsonschema/jsonschema/blob/
    642a09f08318605b16563f47073d3e7b73025029/docs/faq.rst

    License for original code:

    Copyright (c) 2013 Julian Berman

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    return jsonschema.validators.extend(
        validator_class,
        {"properties": set_defaults},
    )


def main():
    cli_args = parse_command_line_arguments()
    if not cli_args.json_config_file.exists():
        raise ValueError(f"No configuration file found at {cli_args.json_config_file}")
    if not cli_args.nifti_dir.exists():
        raise ValueError(f"nifti_dir {cli_args.nifti_dir} does not exist")
    if not cli_args.nifti_dir.is_dir():
        raise ValueError(f"nifti_dir {cli_args.nifti_dir} is not a directory")
    if not cli_args.output_dir.exists():
        os.makedirs(cli_args.output_dir)
    with open(cli_args.json_config_file, "r") as f:
        try:
            hyperparameters = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Configuration file {cli_args.json_config_file} not valid JSON"
            ) from e
    schema = load_config_schema()
    validator = extend_validator_with_default(jsonschema.Draft202012Validator)(schema)
    validator.validate(hyperparameters, schema)
    post_process_hyperparameters(hyperparameters, cli_args)
    torch.multiprocessing.set_start_method("spawn")
    train_model(hyperparameters)


if __name__ == "__main__":
    main()
