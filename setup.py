from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='verydeepvae',
    version='0.1.0',
    description='Very deep variational autoencoder models for 3D images in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/r-gray/3d_very_deep_vae',
    author='Robert Gray',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='pytorch, variational autoencoder, 3d',
    packages=find_packages(),
    python_requires='>=3.7, <3.10',
    install_requires=[
        'h5py',
        'matplotlib',
        'monai',
        'nibabel',
        'numpy',
        'pillow',
        'scipy',
        'scikit-image',
        'torch',
        'torchio',
        'torchvision',
        'tqdm',
    ],
)
