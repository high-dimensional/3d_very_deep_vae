import os
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.cuda.amp as amp
import numpy as np
import csv
import platform
import socket
from ..data_tools import data_handling
from ..misc import visuals
from ..misc import misc
from ..orchestration import run_epoch_hemispheres as run_epoch
from monai.data import DataLoader, Dataset
from ..graphs.vdeepvae_bottom_up_graph_translator import Graph as BottomUpGraph
from ..graphs.vdeepvae_top_down_graph_translator import Graph as TopDownGraph
import monai
import torchio
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import math as maths
import torch.distributed as dist
import nibabel as nib
from ..data_tools.data_transformations import create_data_transformations

"""
Reproducability...
"""
random_seed = 42
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)
random.seed(random_seed)


def main(hyper_params):
    """
    This script coordinates everything!
    """

    if 'recon_folder' not in hyper_params:
        hyper_params['recon_folder'] = os.path.join(hyper_params['current_dir'], 'output')

    if 'mat_file_folder' not in hyper_params:
        hyper_params['mat_file_folder'] = os.path.join(hyper_params['current_dir'], 'output')

    hyper_params['host_name'] = socket.gethostname()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(hyper_params['CUDA_devices'])

    if 'LOCAL_WORLD_SIZE' in os.environ and os.environ['LOCAL_WORLD_SIZE'] is not None:
        # This is a dist.run startup
        local_rank = int(os.environ['LOCAL_RANK'])
        hyper_params['local_rank'] = local_rank
        misc.print_0(hyper_params, "Found local_rank in LOCAL_RANK env var: " + str(local_rank))

        global_rank = int(os.environ['RANK'])
        hyper_params['global_rank'] = global_rank
        misc.print_0(hyper_params, "Found global_rank in RANK env var: " + str(global_rank))

        global_world_size = int(os.environ['WORLD_SIZE'])
        hyper_params['global_world_size'] = global_world_size
        misc.print_0(hyper_params, "Found global_world_size in WORLD_SIZE env var: " + str(global_world_size))

        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        hyper_params['world_size'] = world_size
        misc.print_0(hyper_params, "Found world_size in LOCAL_WORLD_SIZE env var: " + str(world_size))

        master_addr = str(os.environ['MASTER_ADDR'])
        hyper_params['master_addr'] = master_addr
        misc.print_0(hyper_params, "Found master_addr in MASTER_ADDR env var: " + str(master_addr))

        master_port = int(os.environ['MASTER_PORT'])
        hyper_params['master_port'] = master_port
        misc.print_0(hyper_params, "Found master_port in MASTER_ADDR env var: " + str(master_port))
    else:
        if hasattr(hyper_params['args'], 'local_rank'):
            local_rank = hyper_params['args'].local_rank
            hyper_params['local_rank'] = local_rank
            misc.print_0(hyper_params, "Found local_rank in input args: " + str(local_rank))
        else:
            local_rank = hyper_params['local_rank']
            misc.print_0(hyper_params, "Found local_rank in hyper_params: " + str(local_rank))

        world_size = hyper_params['world_size']
        master_addr = hyper_params['master_addr']
        os.environ['MASTER_ADDR'] = master_addr
        master_port = hyper_params['master_port']
        os.environ['MASTER_PORT'] = str(master_port)
        misc.print_0(hyper_params, "Found world_size in hyper_params: " + str(world_size))
        misc.print_0(hyper_params, "Found master_addr in hyper_params: " + str(master_addr))
        misc.print_0(hyper_params, "Found master_port in hyper_params: " + str(master_port))

    os.environ['OMP_NUM_THREADS'] = str(hyper_params['workers_per_process'])

    device = torch.device('cuda', local_rank)
    hyper_params['device'] = device
    device_ids = [local_rank]
    torch.cuda.set_device(device)

    if local_rank > 0:
        hyper_params['verbose'] = False

    if 'threads_per_rank' in hyper_params:
        torch.set_num_threads(hyper_params['threads_per_rank'])
        misc.print_0(hyper_params, f"Setting number of threads to {hyper_params['threads_per_rank']}")
    else:
        misc.print_0(hyper_params, "Setting number of threads to 8")
        torch.set_num_threads(8)

    misc.print_0(hyper_params, "NB: Only output from the rank_0 process is displayed")
    misc.print_0(hyper_params, "Master IP: " + str(hyper_params['master_addr']))
    misc.print_0(hyper_params, "Master Port: " + str(hyper_params['master_port']))
    misc.print_0(hyper_params, "Workers per process: " + str(hyper_params['workers_per_process']))

    if hyper_params['world_size'] > 1:
        misc.print_0(hyper_params, "Waiting for all DDP processes to establish contact...", end="")

    if 'LOCAL_WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
    else:
        torch.distributed.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if hyper_params['world_size'] > 1:
        misc.print_0(hyper_params, "contact established!")

    misc.print_0(hyper_params, "Multiprocessing start method: " + torch.multiprocessing.get_start_method())

    misc.print_0(hyper_params, "Torch version: " + torch.__version__)
    misc.print_0(hyper_params, "Torchio version: " + torchio.__version__)
    misc.print_0(hyper_params, "Numpy version: " + np.__version__)
    misc.print_0(hyper_params, "Monai version: " + monai.__version__)
    misc.print_0(hyper_params, "Torchvision version: " + torchvision.__version__)
    misc.print_0(hyper_params, "Host name: " + hyper_params['host_name'])
    misc.print_0(hyper_params, "CUDA version: " + torch.version.cuda)
    misc.print_0(hyper_params, "CUDA_VISIBLE_DEVICES: " + os.environ['CUDA_VISIBLE_DEVICES'])

    # To make it easier to run code at home, work & on servers, paths for these different environments are hard-coded.
    # Host name + OS is used to determine which set of paths to use.
    if platform.system() == 'Linux':
        if hyper_params['host_name'] == 'robert-work':
            hyper_params['h5py_base_dir'] = '/mnt/wwn-0x5002538d42570f81-part1/DataForNeuralNetworks/'
            hyper_params['nifti_base_dir'] = '/media/robert/Data1/'
            if 'checkpoint_folder' not in hyper_params:
                hyper_params['checkpoint_folder'] = os.path.join('/home/robert/Documents/Torch_Checkpoints/',
                                                                 hyper_params['model_name'])
            if 'tensorboard_dir' not in hyper_params:
                hyper_params['tensorboard_dir'] = os.path.join('/home/robert/Documents/Torch_TensorBoard/',
                                                               hyper_params['model_name'])
        else:
            hyper_params['h5py_base_dir'] = '/local_dir/DataForNeuralNetworks/'
            hyper_params['nifti_base_dir'] = '/local_dir/big_data/'
            if 'checkpoint_folder' not in hyper_params:
                hyper_params['checkpoint_folder'] = os.path.join('/local_dir/Torch_Checkpoints/',
                                                                 hyper_params['model_name'])
            if 'tensorboard_dir' not in hyper_params:
                hyper_params['tensorboard_dir'] = os.path.join('/local_dir/Torch_TensorBoard/',
                                                               hyper_params['model_name'])
    else:
        hyper_params['nifti_base_dir'] = 'C:\\Users\\rober\\Documents\\'
        hyper_params['h5py_base_dir'] = 'C:\\Users\\rober\\Documents\\DataForNeuralNetworks\\'
        if 'checkpoint_folder' not in hyper_params:
            hyper_params['checkpoint_folder'] = os.path.join('C:\\Users\\rober\\Documents\\Torch_Checkpoints\\',
                                                             hyper_params['model_name'])
        if 'tensorboard_dir' not in hyper_params:
            hyper_params['tensorboard_dir'] = os.path.join('C:\\Users\\rober\\Documents\\Torch_TensorBoard\\',
                                                           hyper_params['model_name'])

    samples_dir = os.path.join(hyper_params['recon_folder'], 'samples')

    # Checking folder exist is handled by the rank_0 process only!
    if local_rank == 0:
        if not os.path.exists(hyper_params['tensorboard_dir']):
            os.makedirs(hyper_params['tensorboard_dir'])
        if not os.path.exists(hyper_params['recon_folder']):
            os.makedirs(hyper_params['recon_folder'])
        if not os.path.exists(hyper_params['checkpoint_folder']):
            os.makedirs(hyper_params['checkpoint_folder'])

        writer = SummaryWriter(log_dir=hyper_params['tensorboard_dir'])
        misc.print_0(hyper_params, "To start TB: tensorboard --logdir " + hyper_params['tensorboard_dir'] + " --port=8008")
        if not (hyper_params['resume_from_checkpoint'] and hyper_params['restore_optimiser']):
            data_handling.delete_directory_contents(hyper_params['tensorboard_dir'])
    else:
        writer = None

    misc.print_0(hyper_params, "Checkpoint folder: " + hyper_params['checkpoint_folder'])

    ############################################################################################################

    if hyper_params['use_nii_data']:
        if 'nii_target_shape' in hyper_params:
            data_shape = hyper_params['nii_target_shape']
        else:
            misc.print_0(hyper_params, "You must specify the resample target shape using the data_shape key!")
            quit()

        if hyper_params['resume_from_checkpoint']:
            state_dict_fullpath = os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt')
            misc.print_0(hyper_params, "Resuming from checkpoint: " + state_dict_fullpath)

            if not os.path.exists(state_dict_fullpath):
                misc.print_0(hyper_params, "Checkpoint not found: " + state_dict_fullpath)
                quit()

            checkpoint = torch.load(state_dict_fullpath, map_location="cpu")
    
            if hyper_params['sequence_type'] == 'dwi':
                nifti_b1000_filenames = checkpoint['nifti_b1000_filenames']
                misc.print_0(hyper_params, "Number of niftis: " + str(len(nifti_b1000_filenames)))
                nifti_b1000_paths = [os.path.join(hyper_params['nifti_dwi_dir'], name) for name in nifti_b1000_filenames]
            elif hyper_params['sequence_type'] == 'flair':
                filenames_flair = checkpoint['filenames_flair']
                # filenames_seg = checkpoint['filenames_seg']
                misc.print_0(hyper_params, "Number of niftis: " + str(len(filenames_flair)))
                nifti_paths_flair = [os.path.join(hyper_params['nifti_flair_dir'], name) for name in filenames_flair]
                # nifti_paths_seg = [os.path.join(hyper_params['nifti_flair_dir'], name) for name in filenames_seg]
                nifti_b1000_filenames = filenames_flair
                nifti_b1000_paths = nifti_paths_flair
        else:
            misc.print_0(hyper_params, "Sequence type: " + hyper_params['sequence_type'])
    
            # This is the hemisphere model's nifti loader code
            if misc.key_is_true(hyper_params, 'hack_use_t1'):
                misc.print_0(hyper_params, "HACK: using T1s")
                filenames = os.listdir(hyper_params['nifti_flair_dir'])
                filenames = [f for f in filenames if not f.startswith('.')]  # Remove hidden files
                filenames = [f for f in filenames if '_20253_2_0.zip' in f]  # Remove hidden files
                filenames_flair = [f for f in filenames if '_t1' in f]
            else:
                filenames = os.listdir(hyper_params['nifti_flair_dir'])
                filenames = [f for f in filenames if not f.startswith('.')]  # Remove hidden files
                filenames = [f for f in filenames if '_20253_2_0.zip' in f]  # Remove hidden files
                filenames_flair = [f for f in filenames if '_flair' in f]
    
            nifti_paths_flair = [os.path.join(hyper_params['nifti_flair_dir'], name) for name in filenames_flair]

            eids = [f.split('_')[0] for f in filenames_flair]
    
            if hyper_params['load_metadata']:
                misc.print_0(hyper_params, "Loading metadata and partitioning data into normal/abnormal")
                filename = hyper_params['biobank_eids_dir'] + 'biobank_eids_with_white_matter_hyperintensities.csv'
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
    
                misc.print_0(hyper_params, f"Mean wml vol: {np.mean(wml_volumes)}; std: {np.std(wml_volumes)}")
                wml_mean = np.mean(wml_volumes)
    
                ind_normal = wml_volumes <= wml_mean
                eids_normal = [a for a, b in zip(eids_with_lesions, ind_normal) if b]
                eids_normal = [f for f in eids if f in eids_normal]
    
                ind_abnormal = wml_volumes > wml_mean
                eids_abnormal = [a for a, b in zip(eids_with_lesions, ind_abnormal) if b]
                eids_abnormal = [f for f in eids if f in eids_abnormal]
                misc.print_0(hyper_params, f"Number of normals: {str(len(eids_normal))}; number of abnormals: {str(len(eids_abnormal))}")
    
                filenames_flair = [f for f in filenames_flair if f.split('_')[0] in eids_normal]
                filenames_seg = [f.replace('_flair.nii', '_seg.nii') for f in filenames_flair]
    
                nifti_paths_flair = [os.path.join(hyper_params['nifti_flair_dir'], name) for name in filenames_flair]
                nifti_paths_seg = [os.path.join(hyper_params['nifti_flair_dir'], name) for name in filenames_seg]
    
            nifti_b1000_filenames = filenames_flair
            nifti_b1000_paths = nifti_paths_flair
    
            if 'max_niis_to_use' in hyper_params:
                misc.print_0(hyper_params,
                                     "Restricting to only " + str(hyper_params['max_niis_to_use']) + " niftis")
                filenames_flair = filenames_flair[0:hyper_params['max_niis_to_use']]
                # filenames_seg = filenames_seg[0:hyper_params['max_niis_to_use']]
                nifti_b1000_filenames = nifti_b1000_filenames[0:hyper_params['max_niis_to_use']]
                nifti_paths_flair = nifti_paths_flair[0:hyper_params['max_niis_to_use']]
                # nifti_paths_seg = nifti_paths_seg[0:hyper_params['max_niis_to_use']]
                nifti_b1000_paths = nifti_b1000_paths[0:hyper_params['max_niis_to_use']]
    
                misc.print_0(hyper_params, "B_1000s: " + str(len(nifti_b1000_filenames)))
    
        training_set_size = np.floor(len(nifti_b1000_paths) * hyper_params['train_frac']).astype(np.int32)
        validation_set_size = len(nifti_b1000_paths) - training_set_size
        misc.print_0(hyper_params, "Training niftis: " + str(training_set_size))
        misc.print_0(hyper_params, "Validation niftis: " + str(validation_set_size))
    
        train_files = [{"full_brain": x} for x in zip(nifti_paths_flair[0:training_set_size])]
        val_files = [{"full_brain": x} for x in zip(nifti_paths_flair[training_set_size::])]

        val_transforms, train_transforms = create_data_transformations(hyper_params, device)

        dataset_train = Dataset(data=train_files, transform=train_transforms)
        dataset_val = Dataset(data=val_files, transform=val_transforms)
    
        # pin_memory = True
        pin_memory = False
        loader_train = DataLoader(dataset_train, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True,
                                  num_workers=hyper_params['workers_per_process'], pin_memory=pin_memory)
        misc.print_0(hyper_params, "WARNING: dropping last on val set!")
    
        cardinality_train = len(dataset_train)
        cardinality_val = len(dataset_val)
        hyper_params['cardinality_train'] = cardinality_train
        hyper_params['cardinality_val'] = cardinality_val
    
        is_3d = True
        is_colour = False
    
        batch_count_train = np.ceil(cardinality_train / hyper_params['batch_size']).astype(np.int32)
        batch_count_val = np.ceil(cardinality_val / hyper_params['batch_size']).astype(np.int32)
    
        misc.print_0(hyper_params, "Training set size: " + str(cardinality_train))
        misc.print_0(hyper_params, "Validation set size: " + str(cardinality_val))
        misc.print_0(hyper_params, "Training batches per epoch: " + str(batch_count_train))
        misc.print_0(hyper_params, "Validation batches per epoch: " + str(batch_count_val))
    
        hyper_params['data_shape'] = data_shape
        hyper_params['data_is_3d'] = is_3d
        hyper_params['data_is_colour'] = is_colour

        if local_rank == 0 and not os.path.isdir(samples_dir):
            os.mkdir(samples_dir)
    else:
        print("JPEG directory specified in hyperparameters")
        print("JPEG base dir: " + hyper_params['jpeg_dir'])

        from torchvision import transforms
        from ..data_tools.jpeg_dataset import JPEGDataset

        jpeg_dir = hyper_params['jpeg_dir']
        file_names = [f for f in os.listdir(jpeg_dir) if os.path.isfile(os.path.join(jpeg_dir, f))]
        csv_attr_path = None

        if 'max_jpegs_to_use' in hyper_params:
            file_names = file_names[0:hyper_params['max_jpegs_to_use']]

        train_frac = hyper_params['train_frac']
        print("Creating training/validation set split: proportion for training: " + str(train_frac))
        cardinality = len(file_names)
        cardinality_train = int(train_frac * cardinality)

        if hyper_params['resume_from_checkpoint']:
            # Overwrite the file listings using the lists in the checkpoint
            state_dict_fullpath = os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt')
            misc.print_0(hyper_params, "Resuming from checkpoint: " + state_dict_fullpath)

            if not os.path.exists(state_dict_fullpath):
                misc.print_0(hyper_params, "Checkpoint not found: " + state_dict_fullpath)
                quit()
                
            checkpoint = torch.load(state_dict_fullpath, map_location="cpu")
            file_names_train = checkpoint['file_names_train']
            file_names_val = checkpoint['file_names_val']

            if 'max_jpegs_to_use' in hyper_params:
                # Just in case I change this value and then resume training...
                file_names = file_names[0:hyper_params['max_jpegs_to_use']]

        else:
            file_names_train = file_names[0:cardinality_train]
            file_names_val = file_names[cardinality_train:]

        train_transforms = transforms.Compose([
            # transforms.RandomAffine(3, translate=(0, 0.1), scale=(0.75, 1),
            #                         shear=3, resample=Image.BILINEAR,
            #                         fillcolor=0),
            transforms.Resize(hyper_params['jpeg_target_shape'][0]),
            transforms.CenterCrop(hyper_params['jpeg_target_shape'][0]),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ToTensor()
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(hyper_params['jpeg_target_shape'][0]),
            transforms.CenterCrop(hyper_params['jpeg_target_shape'][0]),
            # transforms.ToTensor()
        ])

        dataset_train = JPEGDataset(jpeg_dir, file_names_train, csv_attr_path, train_transforms, hyper_params)
        dataset_val = JPEGDataset(jpeg_dir, file_names_val, csv_attr_path, val_transforms, hyper_params)

        cardinality_train = len(dataset_train)
        cardinality_val = len(dataset_val)

        is_3d = False
        data_shape = [1] + hyper_params['jpeg_target_shape'] * 2
        is_colour = False
        
        hyper_params['data_shape'] = data_shape
        hyper_params['data_is_3d'] = is_3d
        hyper_params['data_is_colour'] = is_colour

    if 'visualise_training_pipeline_before_starting' in hyper_params and \
            hyper_params['visualise_training_pipeline_before_starting']:
        misc.print_0(hyper_params, "Plotting pipeline before training")
        # for batch in tqdm(loader_train_to_visualise, total=10, desc='Printing augmentations'):
        check_data = next(iter(loader_train))
        paths = check_data['full_brain_meta_dict']['filename_or_obj']
        names = [f.split('/')[-1] for f in paths]
        keys = ["full_brain"]
        to_plot = [check_data[k] for k in keys]
        titles = keys
        current_dir = os.path.join(hyper_params['recon_folder'], 'pipeline_test')
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        num_to_plot = np.min([hyper_params['subjects_to_plot'], hyper_params['batch_size']])

        for k in range(num_to_plot):
            nib.save(nib.Nifti1Image(np.squeeze(to_plot[0][k].cpu().numpy()), np.eye(4)),
                     os.path.join(current_dir, names[k] + "_full_brain" + str(hyper_params['local_rank']) + ".nii.gz"))

        # if not ('use_DDP' in hyper_params and hyper_params['use_DDP'] and hyper_params['local_rank'] > 0):
        visuals.plot_3d_recons_v2(to_plot, titles, None, current_dir,
                                  subjects_to_show=num_to_plot,
                                  hyper_params=hyper_params, prefix=str(hyper_params['local_rank']) + "_")

        misc.print_0(hyper_params, "")

    dataset = [cardinality_train, dataset_train, cardinality_val, dataset_val, data_shape, is_3d, is_colour]

    bottom_up_graph_1 = BottomUpGraph(hyper_params=hyper_params, device=device, input_channels=1)
    bottom_up_graph_2 = None  # Tells the conditional machinery to stand down...
    top_down_graph = TopDownGraph(hyper_params=hyper_params, device=device)

    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=local_rank, shuffle=True,
                                       drop_last=True)
    sampler_val = DistributedSampler(dataset_val, num_replicas=world_size, rank=local_rank, shuffle=False,
                                     drop_last=True)

    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2 ** 32
    #     numpy.random.seed(worker_seed)
    #     random.seed(worker_seed)

    pin_memory = True
    loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=hyper_params['batch_size'],
                              drop_last=True, num_workers=hyper_params['workers_per_process'], pin_memory=pin_memory)
    # loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=hyper_params['batch_size'], drop_last=True,
    #                         num_workers=hyper_params['workers_per_process'], pin_memory=pin_memory)
    loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=hyper_params['batch_size'], drop_last=False,
                            num_workers=hyper_params['workers_per_process'], pin_memory=pin_memory)
    # loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=hyper_params['batch_size'], drop_last=True,
    #                         num_workers=hyper_params['workers_per_process'], worker_init_fn=seed_worker, pin_memory=pin_memory)

    bottom_up_graph_1.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bottom_up_graph_1.model)
    top_down_graph.latents = torch.nn.SyncBatchNorm.convert_sync_batchnorm(top_down_graph.latents)
    top_down_graph.x_mu = torch.nn.SyncBatchNorm.convert_sync_batchnorm(top_down_graph.x_mu)
    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
            hyper_params['predict_x_var']:
        top_down_graph.x_var = torch.nn.SyncBatchNorm.convert_sync_batchnorm(top_down_graph.x_var)

    bottom_up_graph_1.model = torch.nn.parallel.DistributedDataParallel(bottom_up_graph_1.model,
                                                                        device_ids=device_ids,
                                                                        output_device=device)
    top_down_graph.latents = torch.nn.parallel.DistributedDataParallel(top_down_graph.latents,
                                                                       device_ids=device_ids,
                                                                       output_device=device)
    top_down_graph.x_mu = torch.nn.parallel.DistributedDataParallel(top_down_graph.x_mu,
                                                                    device_ids=device_ids,
                                                                    output_device=device)
    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
            hyper_params['predict_x_var']:
        top_down_graph.x_var = torch.nn.parallel.DistributedDataParallel(top_down_graph.x_var,
                                                                         device_ids=device_ids,
                                                                         output_device=device)

    params = []

    if hyper_params['optimise_encoder']:
        misc.print_0(hyper_params, "Optimising encoder")
        params += list(bottom_up_graph_1.model.parameters())

    if hyper_params['optimise_xmu']:
        params += list(top_down_graph.x_mu.parameters())

    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
            hyper_params['predict_x_var']:
        if hyper_params['optimise_xvar']:
            params += list(top_down_graph.x_var.parameters())

    # print("Always optimising the latents")
    # params += list(top_down_graph.latents.parameters())

    if hyper_params['optimise_only_prior']:
        misc.print_0(hyper_params, "Optimising only the prior in the decoder")
        params_sans_prior_predictors = []
        params_prior_predictors = []
        for name, param in top_down_graph.latents.named_parameters():
            if 'convs_p' in name:
                params_prior_predictors.append(param)
            else:
                params_sans_prior_predictors.append(param)
        params += params_prior_predictors
        misc.print_0(hyper_params, "Parameters in prior being optimised: " + str(misc.count_parameters(params_prior_predictors)))
    elif hyper_params['optimise_prior']:
        misc.print_0(hyper_params, "Optimising the prior")
        params += list(top_down_graph.latents.parameters())
        params_sans_prior_predictors = None
    else:
        misc.print_0(hyper_params, "Not optimising the prior")
        params_sans_prior_predictors = []
        params_prior_predictors = []
        for name, param in top_down_graph.latents.named_parameters():
            if 'convs_p' in name:
                params_prior_predictors.append(param)
            else:
                params_sans_prior_predictors.append(param)
        params += params_sans_prior_predictors
        misc.print_0(hyper_params, "Ommitted parameters in prior: " + str(misc.count_parameters(params_prior_predictors)))

    # params_sans_latent_predictors = []
    # for name, param in top_down_graph.latents.named_parameters():
    #     if 'convs_p' not in name and 'convs_q' not in name:
    #         params_sans_latent_predictors.append(param)
    # params += params_sans_latent_predictors
    # params += top_down_graph.posterior_params
    # params += top_down_graph.prior_params
    # print("")

    # if 'latents_to_optimise' in hyper_params:
    #     misc.print_0(hyper_params, "Optimising latents selectively")
    #
    #     # First split the params into those that predict the latents and those that don't
    #     params_sans_latent_predictors = []
    #     params_prior_predictor_names = []
    #     params_prior_predictor_params = []
    #     params_posterior_predictor_names = []
    #     params_posterior_predictor_params = []
    #
    #     for name, param in top_down_graph.latents.named_parameters():
    #         if 'convs_p' in name:
    #             params_prior_predictor_names.append(name)
    #             params_prior_predictor_params.append(param)
    #         elif 'convs_q' in name:
    #             params_posterior_predictor_names.append(name)
    #             params_posterior_predictor_params.append(param)
    #         else:
    #             params_sans_latent_predictors.append(param)
    #
    #     params += params_sans_latent_predictors
    #
    #     # Now selectively add the params that
    #     for k, val in enumerate(hyper_params['latents_to_optimise']):
    #         pass

    misc.print_0(hyper_params, "Parameters in bottom-up graph 1: " + str(
        misc.count_unique_parameters(list(bottom_up_graph_1.model.named_parameters()))))
    misc.print_0(hyper_params, "Parameters in top-down graph: " + str(
        misc.count_unique_parameters(list(top_down_graph.latents.named_parameters()))))
    misc.print_0(hyper_params,
        "Parameters in x_mu graph: " + str(misc.count_unique_parameters(list(top_down_graph.x_mu.named_parameters()))))
    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
            hyper_params['predict_x_var']:
        misc.print_0(hyper_params, "Parameters in x_var graph: " + str(
            misc.count_unique_parameters(list(top_down_graph.x_var.named_parameters()))))
    misc.print_0(hyper_params, "Total number of trainable parameters: " + str(misc.count_parameters(params)))

    if hyper_params['print_model']:
        print_0(bottom_up_graph_1.model)
        print_0(top_down_graph.latents)
        print_0(top_down_graph.x_mu)
        if hyper_params['separate_loc_scale_convs'] and hyper_params['predict_x_var']:
            print_0(top_down_graph.x_var)

    # optimizer = optim.Adam(list(params), lr=hyper_params['learning_rate'])
    optimizer = optim.Adamax(list(params), lr=hyper_params['learning_rate'])

    scaler = amp.GradScaler(enabled=hyper_params['half_precision'])
    if hyper_params['half_precision']:
        misc.print_0(hyper_params, "Using AMP-based mixed precision")
        # scaler = amp.GradScaler()

    if hyper_params['resume_from_checkpoint']:
        misc.print_0(hyper_params, "Resuming from checkpoint")
        state_dict_fullpath = os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt')
        checkpoint = torch.load(state_dict_fullpath, map_location='cpu')

        bottom_up_graph_1.model.load_state_dict(checkpoint['bottom_up_graph_1_state_dict'])
        # top_down_graph.latents.load_state_dict(checkpoint['top_down_generative_graph_state_dict'])
        top_down_graph.latents.load_state_dict(checkpoint['top_down_generative_graph_state_dict'], strict = False)
        top_down_graph.x_mu.load_state_dict(checkpoint['top_down_x_mu_graph_state_dict'])
        if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
                hyper_params['predict_x_var']:
            top_down_graph.x_var.load_state_dict(checkpoint['top_down_x_var_graph_state_dict'])

        validation_mask = checkpoint['validation_mask']  # Still used?

        if 'kl_weight_auto_adjustment' in hyper_params and hyper_params['kl_weight_auto_adjustment']:
            kl_weights = checkpoint['kl_weights']
        else:
            kl_weights = None

        if hyper_params['restore_optimiser']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if hyper_params['half_precision']:
                scaler.load_state_dict(checkpoint['scaler'])
            starting_epoch = checkpoint['epoch'] + 1

            loss_history_train_kl = checkpoint['loss_history_train_kl']
            loss_history_train_kl_all = checkpoint['loss_history_train_kl_all']
            loss_history_train_mse = checkpoint['loss_history_train_mse']
            loss_history_train_loss = checkpoint['loss_history_train_loss']
            loss_history_train_nll_bits_per_dim = checkpoint['loss_history_train_nll_bits_per_dim']
            loss_history_val_kl = checkpoint['loss_history_val_kl']
            loss_history_val_kl_all = checkpoint['loss_history_val_kl_all']
            loss_history_val_mse = checkpoint['loss_history_val_mse']
            loss_history_val_loss = checkpoint['loss_history_val_loss']
            loss_history_val_nll_bits_per_dim = checkpoint['loss_history_val_nll_bits_per_dim']
            
            misc.print_0(hyper_params, "Previous (approximate) train loss: " + str(loss_history_train_loss[-1][1]))
            misc.print_0(hyper_params, "Previous (approximate) validation loss: " + str(loss_history_val_loss[-1][1]))

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

        if 'kl_weight_auto_adjustment' in hyper_params and hyper_params['kl_weight_auto_adjustment']:
            kl_weights = torch.ones((sum(hyper_params['latents_per_channel']), 1))
        else:
            kl_weights = None

    if 'use_latent_weight_file' in hyper_params and hyper_params['use_latent_weight_file']:
        misc.print_0(hyper_params, "Using a KL weight file")
        if hyper_params['create_latent_weight_file']:
            misc.print_0(hyper_params, "Creating KL weight file: " + hyper_params['latent_weight_file_location'])
            with open(hyper_params['latent_weight_file_location'], mode='w') as weight_file:
                weight_writer = csv.writer(weight_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                row = []
                for k in range(sum(hyper_params['latents_per_channel'])):
                    row.append(str(k) + ':1')
                weight_writer.writerow(row)

        with open(hyper_params['latent_weight_file_location']) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                pass
            kl_weights = [float(a.split(':')[1]) for a in row]  # New!
        misc.print_0(hyper_params, "KL weights: " + str(kl_weights))

    if misc.key_is_true(hyper_params, 'export_progressive_vols'):
        misc.print_0(hyper_params, "Exporting progressive recons as niftis")

        output_dir = '/media/robert/Data2/Progressively_encoded_data/'
        misc.print_0(hyper_params, "Exporting progressively encoded data")

        with torch.no_grad():
            bottom_up_graph_1.model.eval()
            top_down_graph.latents.eval()
            top_down_graph.x_mu.eval()
            if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
                    hyper_params['predict_x_var']:
                top_down_graph.x_var.eval()

            # Tell the top down block which resolutions it should sample from the posterior
            # tqdm_obj = misc.tqdm_on_rank_0(hyper_params,
            #                                range(1, 1 + len(hyper_params['channels_per_latent'])),
            #                                desc='Progressive recons')
            # for min in tqdm_obj:
            for min in range(1, 1 + len(hyper_params['channels_per_latent'])):
                max = len(hyper_params['channels_per_latent'])
                res_to_sample_from_prior = [2 ** p for p in range(min, max)]

                torch.cuda.empty_cache()

                with amp.autocast(hyper_params['half_precision']):

                    for loader in [loader_val, loader_train]:

                        for batch in tqdm(loader):
                            # batch = next(iter(loader_val))

                            if hyper_params['use_nii_data']:
                                # In this case use dictionaries
                                current_input = batch['full_brain'].to(device)
                            else:
                                current_input = batch[0].to(device)

                            input_dictionary_1 = {'data': current_input}

                            data_dictionary_1 = bottom_up_graph_1.model(input_dictionary_1)

                            data_dictionary = {'data': data_dictionary_1['data'], 'KL_list': [],
                                               'res_to_sample_from_prior': res_to_sample_from_prior}
                            for key in data_dictionary_1:
                                data_dictionary['encoder1_' + key] = data_dictionary_1[key]

                            data_dictionary_latents = top_down_graph.latents(data_dictionary)
                            data_dictionary_x_mu = top_down_graph.x_mu(data_dictionary_latents)

                            x_mu, x_std, x_var, x_log_var = misc.gaussian_output(data_dictionary_x_mu, top_down_graph,
                                                                                 hyper_params, num_modalities=1)

                            if hyper_params['half_precision']:
                                x_mu = x_mu.type(torch.float32)
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    x_std = x_std.type(torch.float32)

                            x_mu = x_mu.cpu().detach().numpy()
                            if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                x_std = x_std.cpu().detach().numpy()

                            if hyper_params['use_nii_data']:
                                file_paths = batch['full_brain_meta_dict']['filename_or_obj']
                                file_names = [os.path.basename(a).split("/")[0] for a in file_paths]

                            if not os.path.exists(output_dir + 'res_' + str(min) + '/'):
                                os.makedirs(output_dir + 'res_' + str(min) + '/')

                            for k in range(len(file_paths)):
                                current_vol = np.squeeze(x_mu[k], 0)
                                current_name = file_names[k]

                                nifti = nib.Nifti1Image(current_vol, np.eye(4))
                                nib.save(nifti, output_dir + 'res_' + str(min) + '/' + current_name)

        quit()

    for epoch in range(starting_epoch, hyper_params['total_epochs']):
        torch.cuda.empty_cache()  # Reduces risk of memory leaks after repeated computation of recons, samples etc

        if 'h5py_filename' in hyper_params:  # For the h4f5 data, I need to reset the loader...
            loader_train = dataset.data_loader(dataset.cardinality_train, dataset.dataset_train,
                                               hyper_params['batch_size'])

        sampler_train.set_epoch(epoch)  # Shuffle each epoch

        epoch_dict = {'bottom_up_graph_1': bottom_up_graph_1,
                      'bottom_up_graph_2': bottom_up_graph_2,
                      'top_down_graph': top_down_graph,
                      'hyper_params': hyper_params,
                      'optimizer': optimizer,
                      'device_ids': device_ids,
                      'validation_mask': None,
                      # 'params_sans_prior_predictors': params_sans_prior_predictors,
                      'params_sans_prior_predictors': None,
                      'kl_weights': kl_weights,
                      'loader': loader_train,
                      'scaler': scaler,
                      'params': params,
                      'epoch': epoch,
                      'writer': writer,
                      'training': True,
                      'progress_bar_text': 'Optimising',
                      'summary_text_prefix': "Approx loss",
                      'writer_prefix': "Training",
                      'device': device
                      }
        output_dict = run_epoch.go(epoch_dict)

        if 'kl_weight_auto_adjustment' in hyper_params and hyper_params['kl_weight_auto_adjustment']:
            kl_weights = output_dict['kl_weights']
        else:
            kl_weights = None

        loss_tally_train_kl = output_dict['kl_tally']
        loss_tally_train_kl_all = output_dict['kl_all_tallies']
        loss_tally_train_mse = output_dict['mse_tally']
        loss_tally_train_loss = output_dict['loss_tally']
        loss_tally_train_nll_per_dim = output_dict['nll_tally_in_bits_per_dim']

        loss_history_train_kl.append([epoch, loss_tally_train_kl])

        for key in loss_tally_train_kl_all:
            if key in loss_history_train_kl_all:
                loss_history_train_kl_all[key].append([epoch, loss_tally_train_kl_all[key]])
            else:
                loss_history_train_kl_all[key] = [[epoch, loss_tally_train_kl_all[key]]]

        loss_history_train_mse.append([epoch, loss_tally_train_mse])
        loss_history_train_loss.append([epoch, loss_tally_train_loss])
        loss_history_train_nll_bits_per_dim.append([epoch, loss_tally_train_nll_per_dim])

        if not hyper_params['validation_period'] == 1 and hyper_params['local_rank'] == 0:
            visuals.plot_error_curves(data=[loss_history_train_loss], labels=['training error'],
                                      plot_title='Training error', recon_folder=hyper_params['recon_folder'],
                                      prefix='loss_train')
            visuals.plot_error_curves(data=[loss_history_train_kl], labels=['training KL'],
                                      plot_title='Training KL', recon_folder=hyper_params['recon_folder'],
                                      prefix='kl_train')
            visuals.plot_error_curves(data=[loss_history_train_mse], labels=['training MSE'],
                                      plot_title='Training MSE', recon_folder=hyper_params['recon_folder'],
                                      prefix='mse_train')
            visuals.plot_error_curves(data=[loss_history_train_nll_bits_per_dim], labels=['training log likelihood/dim'],
                                      plot_title='Training log likelihood/dim', recon_folder=hyper_params['recon_folder'],
                                      prefix='nll_per_dim_train')

        # Plot the gradient norms
        if hyper_params['local_rank'] == 0 and 'plot_gradient_norms' in hyper_params and hyper_params['plot_gradient_norms']:
            grad_norms = output_dict['gradient_norms']
            keys = list(grad_norms.keys())
            data = [grad_norms[k] for k in keys]
            visuals.plot_error_curves(data=data, labels=keys,
                                      plot_title='Training gradient norms',
                                      recon_folder=hyper_params['recon_folder'],
                                      prefix='grad_norms_train',
                                      xlabel="Iteration",
                                      precision=3)

        del output_dict
        torch.cuda.empty_cache()

        if epoch % hyper_params['validation_period'] == 0:
            with torch.no_grad():

                validation_mask = None
                epoch_dict = {'bottom_up_graph_1': bottom_up_graph_1,
                              'bottom_up_graph_2': bottom_up_graph_2,
                              'top_down_graph': top_down_graph,
                              'dataset': dataset,
                              'hyper_params': hyper_params,
                              'optimizer': optimizer,
                              'validation_mask': validation_mask,
                              'loader': loader_val,
                              'scaler': scaler,
                              'params': params,
                              'kl_weights': kl_weights,
                              'epoch': epoch,
                              'writer': writer,
                              'training': False,
                              'progress_bar_text': 'Validating',
                              'summary_text_prefix': "Validation",
                              'writer_prefix': "Validation",
                              'device': device
                              }
                output_dict = run_epoch.go(epoch_dict)

                loss_tally_val_kl = output_dict['kl_tally']
                loss_tally_val_kl_all = output_dict['kl_all_tallies']
                loss_tally_val_mse = output_dict['mse_tally']
                loss_tally_val_loss = output_dict['loss_tally']
                loss_tally_val_nll_per_dim = output_dict['nll_tally_in_bits_per_dim']

                loss_history_val_kl.append([epoch, loss_tally_val_kl])

                for key in loss_tally_val_kl_all:
                    if key in loss_history_val_kl_all:
                        loss_history_val_kl_all[key].append([epoch, loss_tally_val_kl_all[key]])
                    else:
                        loss_history_val_kl_all[key] = [[epoch, loss_tally_val_kl_all[key]]]

                loss_history_val_mse.append([epoch, loss_tally_val_mse])
                loss_history_val_loss.append([epoch, loss_tally_val_loss])
                loss_history_val_nll_bits_per_dim.append([epoch, loss_tally_val_nll_per_dim])

                if hyper_params['local_rank'] == 0:
                    if len(loss_history_val_loss) == len(loss_history_train_loss):
                        visuals.plot_error_curves(data=[loss_history_train_loss, loss_history_val_loss],
                                                  labels=['training error', 'validation error'],
                                                  plot_title='Train & validation error',
                                                  recon_folder=hyper_params['recon_folder'],
                                                  prefix='loss')
                        visuals.plot_error_curves(data=[loss_history_train_kl, loss_history_val_kl],
                                                  labels=['training KL', 'validation KL'],
                                                  plot_title='Train & validation KL',
                                                  recon_folder=hyper_params['recon_folder'],
                                                  prefix='kl')

                        visuals.plot_error_curves(data=[loss_history_train_mse, loss_history_val_mse],
                                                  labels=['training MSE', 'validation MSE'],
                                                  plot_title='Train & validation MSE',
                                                  recon_folder=hyper_params['recon_folder'],
                                                  prefix='mse')
                        visuals.plot_error_curves(data=[loss_history_train_nll_bits_per_dim, loss_history_val_nll_bits_per_dim],
                                                  labels=['training log likelihood/dim', 'validation log likelihood/dim'],
                                                  plot_title='Train & validation log likelihood/dim',
                                                  recon_folder=hyper_params['recon_folder'],
                                                  prefix='nll_per_dim')
                    else:
                        visuals.plot_error_curves(data=[loss_history_val_loss], labels=['validating error'],
                                                  plot_title='Validation error', recon_folder=hyper_params['recon_folder'],
                                                  prefix='loss_val')
                        visuals.plot_error_curves(data=[loss_history_val_kl], labels=['validating KL'],
                                                  plot_title='Validation KL', recon_folder=hyper_params['recon_folder'],
                                                  prefix='kl_val')
                        visuals.plot_error_curves(data=[loss_history_val_mse], labels=['validating MSE'],
                                                  plot_title='Validation MSE', recon_folder=hyper_params['recon_folder'],
                                                  prefix='mse_val')
                        visuals.plot_error_curves(data=[loss_history_val_nll_bits_per_dim], labels=['validating log likelihood/dim'],
                                                  plot_title='Validation log likelihood/dim', recon_folder=hyper_params['recon_folder'],
                                                  prefix='nll_per_dim_val')

                    # Plot the separate KLs
                    dimensionalities = [((2 ** p) ** 3) * q for p, q in
                                        zip(range(len(hyper_params['channels_per_latent'])),
                                            hyper_params['channels_per_latent'][::-1])]
                    dims_per_latent = []
                    for k, dims in enumerate(dimensionalities):
                        num_latents = hyper_params['latents_per_channel'][-1 - k]
                        dims_per_latent += [dims] * num_latents  # Times to repeat this dims

                    keys = list(loss_history_train_kl_all.keys())
                    data = [loss_history_train_kl_all[k] for k in keys]
                    visuals.plot_error_curves(data=data, labels=keys,
                                              plot_title='Training KLs per resolution',
                                              recon_folder=hyper_params['recon_folder'],
                                              prefix='kl_all_train',
                                              precision=5)

                    data_normed = data[:]
                    for k in range(len(data_normed)):
                        current_dims = dims_per_latent[k]
                        data_normed[k] = [[a[0], a[1] / current_dims] for a in data_normed[k]]
                    visuals.plot_error_curves(data=data_normed, labels=keys,
                                              plot_title='Training KLs per dimension, per resolution',
                                              recon_folder=hyper_params['recon_folder'],
                                              prefix='kl_all_train_normed',
                                              precision=8)

                    keys = list(loss_history_val_kl_all.keys())
                    data = [loss_history_val_kl_all[k] for k in keys]
                    visuals.plot_error_curves(data=data, labels=keys,
                                              plot_title='Validation KLs per resolution',
                                              recon_folder=hyper_params['recon_folder'],
                                              prefix='kl_all_val',
                                              precision=5)
                    data_normed = data[:]
                    for k in range(len(data_normed)):
                        current_dims = dims_per_latent[k]
                        data_normed[k] = [[a[0], a[1] / current_dims] for a in data_normed[k]]
                    visuals.plot_error_curves(data=data_normed, labels=keys,
                                              plot_title='Validation KLs per dimension, per resolution',
                                              recon_folder=hyper_params['recon_folder'],
                                              prefix='kl_all_val_normed',
                                              precision=8)

        if hyper_params['save_period'] > 0 and epoch % hyper_params['save_period'] == 0:
            if hyper_params['local_rank'] == 0:
                misc.print_0(hyper_params, "Saving model")
                validation_mask = None
                checkpoint_dict = {'hyper_params': hyper_params,
                                   'epoch': epoch,
                                   'validation_mask': validation_mask,
                                   'bottom_up_graph_1_state_dict': bottom_up_graph_1.model.state_dict(),
                                   'top_down_generative_graph_state_dict': top_down_graph.latents.state_dict(),
                                   'top_down_x_mu_graph_state_dict': top_down_graph.x_mu.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict(),
                                   'loss_history_train_kl': loss_history_train_kl,
                                   'loss_history_train_kl_all': loss_history_train_kl_all,
                                   'loss_history_train_mse': loss_history_train_mse,
                                   'loss_history_train_loss': loss_history_train_loss,
                                   'kl_weights': kl_weights,
                                   'loss_history_train_nll_bits_per_dim': loss_history_train_nll_bits_per_dim,
                                   'loss_history_val_kl': loss_history_val_kl,
                                   'loss_history_val_kl_all': loss_history_val_kl_all,
                                   'loss_history_val_mse': loss_history_val_mse,
                                   'loss_history_val_loss': loss_history_val_loss,
                                   'loss_history_val_nll_bits_per_dim': loss_history_val_nll_bits_per_dim}

                if 'kl_multiplier' in hyper_params:
                    checkpoint_dict['kl_multiplier'] = hyper_params['kl_multiplier']

                if hyper_params['use_nii_data']:
                    if hyper_params['sequence_type'] == 'dwi':
                        checkpoint_dict['nifti_b1000_filenames'] = nifti_b1000_filenames
                    elif hyper_params['sequence_type'] == 'flair':
                        checkpoint_dict['filenames_flair'] = filenames_flair
                        # checkpoint_dict['filenames_seg'] = filenames_seg
                else:
                    checkpoint_dict['file_names_train'] = file_names_train
                    checkpoint_dict['file_names_val'] = file_names_val
                    checkpoint_dict['file_names'] = file_names

                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
                        hyper_params['predict_x_var']:
                    checkpoint_dict['top_down_x_var_graph_state_dict'] = top_down_graph.x_var.state_dict()
                if hyper_params['half_precision']:
                    checkpoint_dict['scaler'] = scaler.state_dict()

                if hyper_params['keep_every_checkpoint']:
                    torch.save(checkpoint_dict, os.path.join(hyper_params['checkpoint_folder'],
                                                             'state_dictionary_' + str(epoch) + '.pt'))

                torch.save(checkpoint_dict, os.path.join(hyper_params['checkpoint_folder'], 'state_dictionary.pt'))

        if epoch % hyper_params['plot_recons_period'] == 0:
            misc.print_0(hyper_params, "Computing training set reconstructions")
            with torch.no_grad():
                bottom_up_graph_1.model.eval()
                top_down_graph.latents.eval()
                top_down_graph.x_mu.eval()
                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['separate_output_loc_scale_convs'] and \
                        hyper_params['predict_x_var']:
                    top_down_graph.x_var.eval()

                torch.cuda.empty_cache()

                with amp.autocast(hyper_params['half_precision']):
                    batch = next(iter(loader_val))

                    if hyper_params['use_nii_data']:
                        # In this case use dictionaries
                        current_input = batch['full_brain'].to(device)
                    else:
                        current_input = batch[0].to(device)

                    if hyper_params['half_precision']:
                        current_input = current_input.type(torch.float32)

                    current_input = current_input.cpu().detach().numpy()

                    to_plot = [current_input]
                    titles = ["input: current_input"]

                    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                        to_plot_std = [current_input]
                        titles_std = ["input: current_input"]

                    # Tell the top down block which resolutions it should sample from the posterior
                    for min in range(1, 1 + len(hyper_params['channels_per_latent'])):
                        max = len(hyper_params['channels_per_latent'])

                        if 'hidden_spatial_dims' in hyper_params:
                            if hyper_params['use_nii_data']:
                                temp = hyper_params['nii_target_shape'][0:1] + hyper_params['hidden_spatial_dims'][:]
                            else:
                                temp = hyper_params['jpeg_target_shape'] + hyper_params['hidden_spatial_dims'][:]

                            res_to_sample_from_prior = temp[::-1][min:]
                        else:
                            res_to_sample_from_prior = [2**p for p in range(min, max)]

                        if hyper_params['use_nii_data']:
                            # In this case use dictionaries
                            current_input = batch['full_brain'].to(device)
                        else:
                            current_input = batch[0].to(device)

                        input_dictionary_1 = {'data': current_input}

                        data_dictionary_1 = bottom_up_graph_1.model(input_dictionary_1)

                        data_dictionary = {'data': data_dictionary_1['data'], 'KL_list': [],
                                           'res_to_sample_from_prior': res_to_sample_from_prior}
                        for key in data_dictionary_1:
                            data_dictionary['encoder1_' + key] = data_dictionary_1[key]

                        data_dictionary_latents = top_down_graph.latents(data_dictionary)
                        data_dictionary_x_mu = top_down_graph.x_mu(data_dictionary_latents)

                        if hyper_params['likelihood'] == 'Gaussian':
                            x_mu, x_std, x_var, x_log_var = misc.gaussian_output(data_dictionary_x_mu, top_down_graph,
                                                                                 hyper_params, num_modalities=1)
                        else:
                            logits = data_dictionary_x_mu['data']
                            probs = logits.detach().cpu()
                            probs = torch.sigmoid(probs)
                            preds = probs.detach().clone()
                            preds[preds >= 0.5] = 1
                            preds[preds < 0.5] = 0

                            x_mu = preds

                        if hyper_params['half_precision']:
                            x_mu = x_mu.type(torch.float32)
                            if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                x_std = x_std.type(torch.float32)

                        x_mu = x_mu.cpu().detach().numpy()
                        if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                            x_std = x_std.cpu().detach().numpy()

                        to_plot += [x_mu]
                        if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                            to_plot_std += [x_std]

                        if min == max:
                            if is_3d:
                                titles += ["E[p(current_input | z)]. No imputation!"]
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    titles_std += ["STD[p(current_input | z)]. No imputation!"]
                            else:
                                titles += ["E[p(current_input | z)].\nNo imputation!"]
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    titles_std += ["STD[p(current_input | z)].\nNo imputation!"]
                        else:
                            if 'hidden_spatial_dims' in hyper_params:
                                res = str(hyper_params['hidden_spatial_dims'][::-1][min-1])
                            else:
                                res = str(2 ** (min - 1))

                            if is_3d:
                                titles += ["E[p(current_input | z)]. Imputing latents above " +
                                           res + " cubed using the prior"]
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    titles_std += ["STD[p(current_input | z)]. Imputing latents above " +
                                           res + " cubed using the prior"]
                            else:
                                titles += ["E[p(current_input | z)].\nImputing latents above " +
                                           res + "\nsquared using the prior"]
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    titles_std += ["STD[p(current_input | z)].\nImputing latents above " +
                                           res + "\nsquared using the prior"]

                    subjects_to_plot_per_process = maths.floor(hyper_params['subjects_to_plot'])
                    prefix = 'prog_recons_rank' + str(local_rank)
                    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                        prefix_std = 'prog_stds_rank' + str(local_rank)
                    if is_3d:
                        visuals.plot_3d_recons_v2(to_plot, titles, epoch, hyper_params['recon_folder'],
                                                  subjects_to_show=subjects_to_plot_per_process,
                                                  hyper_params=hyper_params, prefix=prefix)
                        if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                            visuals.plot_3d_recons_v2(to_plot_std, titles_std, epoch, hyper_params['recon_folder'],
                                                      subjects_to_show=subjects_to_plot_per_process,
                                                      hyper_params=hyper_params, prefix=prefix_std)
                    else:
                        visuals.plot_2d(to_plot, titles, epoch, hyper_params['recon_folder'], filename=prefix,
                                        is_colour=False, num_to_plot=subjects_to_plot_per_process,
                                        norm_recons=True)
                        if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                            visuals.plot_2d(to_plot_std, titles_std, epoch, hyper_params['recon_folder'],
                                            filename=prefix_std,
                                            is_colour=False, num_to_plot=subjects_to_plot_per_process,
                                            norm_recons=True)

                    # Repeat this but now just plot differences
                    subjects_to_plot_per_process = maths.floor(hyper_params['subjects_to_plot'])
                    prefix = 'prog_diffs_rank' + str(local_rank)
                    if is_3d:
                        to_plot = [to_plot[0]] + [np.abs(a - to_plot[0]) for a in to_plot[1:]]
                        visuals.plot_3d_recons_v2(to_plot, titles, epoch, hyper_params['recon_folder'],
                                                  subjects_to_show=subjects_to_plot_per_process,
                                                  hyper_params=hyper_params, prefix=prefix)
                    else:
                        # to_plot = [to_plot[0]] + [np.abs(a - to_plot[0]) for a in to_plot[1:]]
                        visuals.plot_2d(to_plot, titles, epoch, hyper_params['recon_folder'], filename=prefix,
                                        is_colour=False, num_to_plot=subjects_to_plot_per_process,
                                        norm_recons=True)

                    #############################################################################################
                    #############################################################################################
                    #############################################################################################
                    #############################################################################################

                    # Now we create samples!
                    misc.print_0(hyper_params, "\nComputing samples")
                    if not is_3d:
                        times_to_sample = 1  # In 2D, we compute all samples in one forward pass (per GPU)
                    elif 'times_to_sample' in hyper_params:
                        times_to_sample = hyper_params['times_to_sample']
                    else:
                        times_to_sample = 5

                    for temp in [1]:
                        temp_prefix = f"temp{temp}_"

                        for k in range(times_to_sample):
                            data_dictionary = {'data': data_dictionary_1['data'], 'KL_list': []}
                            for key in data_dictionary:
                                if 'encoder1_' in key:
                                    data_dictionary[key] = None

                            data_dictionary['sampling_noise_std_override'] = temp
                            data_dictionary_latents = top_down_graph.latents(data_dictionary)
                            data_dictionary_x_mu = top_down_graph.x_mu(data_dictionary_latents)

                            if hyper_params['likelihood'] == 'Gaussian':
                                samples, samples_std, samples_var, samples_log_var = \
                                    misc.gaussian_output(data_dictionary_x_mu, top_down_graph,
                                                         hyper_params, num_modalities=1)
                            else:
                                logits = data_dictionary_x_mu['data']
                                probs = logits.detach().cpu()
                                probs = torch.sigmoid(probs)
                                preds = probs.detach().clone()
                                preds[preds >= 0.5] = 1
                                preds[preds < 0.5] = 0

                                samples = preds

                            samples = samples.type(torch.float32)
                            samples = samples.cpu().detach().numpy()
                            if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                samples_std = samples_std.type(torch.float32)
                                samples_std = samples_std.cpu().detach().numpy()

                            if is_3d:
                                if local_rank == 0 and not os.path.isdir(samples_dir):
                                    os.mkdir(samples_dir)

                                batch_shape = list(samples.shape)
                                affine = batch['full_brain_meta_dict']['affine'][0].cpu().detach().numpy()

                                for m in range(batch_shape[0]):
                                    current_filename = 'rank' + str(hyper_params['local_rank']) + '_example' + str(m) + '.nii'
                                    current_vol = np.squeeze(samples[m])
                                    nim = nib.Nifti1Image(current_vol, affine)
                                    nib.save(nim, os.path.join(samples_dir, current_filename))

                                to_plot = [samples]
                                titles = ["sample from p(current_input | z)"]
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    to_plot += [samples_std]
                                    titles += ["std of p(current_input | z)"]

                                subjects_to_plot_per_process = np.min([maths.floor(hyper_params['subjects_to_plot']),
                                                                       hyper_params['batch_size']])
                                prefix = temp_prefix + 'samples_' + str(k) + '_'

                                current_dir = [os.path.join(samples_dir, 'rank' + str(local_rank) + '_' + str(n)) for n in
                                               range(subjects_to_plot_per_process)]
                                for d in current_dir:
                                    if not os.path.isdir(d):
                                        os.mkdir(d)

                                visuals.plot_3d_recons_v2(to_plot, titles, epoch, current_dir,
                                                          subjects_to_show=subjects_to_plot_per_process,
                                                          hyper_params=hyper_params, prefix=prefix)

                            else:
                                visuals.image_grid(samples, epoch, hyper_params['recon_folder'],
                                                   temp_prefix + 'samples_rank' + '_' + str(local_rank), 5,
                                                   norm_recons=True)
                                if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
                                    visuals.image_grid(samples_std, epoch, hyper_params['recon_folder'],
                                                       temp_prefix + 'samples_stds_rank' + '_' + str(local_rank), 5,
                                                       norm_recons=True)

                            del data_dictionary
                            del data_dictionary_latents
                            del data_dictionary_x_mu
                            if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var'] and \
                                    hyper_params['separate_output_loc_scale_convs']:
                                del data_dictionary_log_var

                    # data_dictionary_1 is left over from the recon logic and is used each time we sample
                    del data_dictionary_1

    writer.close()


if __name__ == '__main__':
    freeze_support()
    main()
