"""Generate NIfTI images as synthetic training data set."""

import argparse
import os
import pathlib
import numpy as np
import nibabel as nb


def generate_synthetic_image(rng, args):
    return rng.uniform(0, 1, size=(args.voxels_per_axis,) * 3)


def main():
    parser = argparse.ArgumentParser(
        "Generate NIfTI volumetric images as synthetic training data set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--voxels_per_axis",
        type=int,
        default=32,
        help="The number of voxels along each of the three spatial axes",
    )
    parser.add_argument(
        "--number_of_files",
        type=int,
        default=1000,
        help="The number of NIfTI files to generate",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Integer seed to use to initialise pseudo-random number generator state",
    )
    parser.add_argument(
        "--output_directory",
        type=pathlib.Path,
        required=True,
        help="The directory to write the generated files to",
    )
    args = parser.parse_args()
    affine_transformation = np.identity(4)
    if not args.output_directory.exists():
        os.makedirs(args.output_directory)
    if args.number_of_files <= 0:
        raise ValueError("number_of_files must be a positive integer")
    if args.voxels_per_axis <= 0 or (np.log2(args.voxels_per_axis) % 1) != 0.0:
        raise ValueError("voxels_per_image must be a positive power of two")
    rng = np.random.default_rng(args.random_seed)
    for i in range(args.number_of_files):
        voxels = generate_synthetic_image(rng, args)
        nifti_image = nb.Nifti1Image(voxels, affine_transformation)
        nb.save(nifti_image, args.output_directory / f"generated_{i}.nii")


if __name__ == "__main__":
    main()
