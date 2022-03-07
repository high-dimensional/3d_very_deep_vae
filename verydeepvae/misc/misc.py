import torch
import torch.nn as nn
import numpy as np
import h5py
from scipy.spatial.distance import pdist
from tqdm import tqdm
import torch.distributed as dist
import itertools
import random
import re
import operator as op
from functools import reduce


def dice(preds, labels):
    dims = non_batch_dims(preds)
    numerator = 2 * torch.sum(preds * labels, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(labels, dim=dims)
    sorenson_dice = torch.mean(numerator / denominator)

    return sorenson_dice


def ncr(n, r):
    # Copied from https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def sort_dist(event, num_imaging_modalities):
    if '|' in event:
        left_of_pipe, right_of_pipe = event.split('|')
    else:
        left_of_pipe, right_of_pipe = event, ''

    def sort_one_side(x):
        x = make_one_side_of_pipe_numeric(x, num_imaging_modalities)
        x = x.split(',')
        x = [int(x) for x in x]
        x.sort()
        for k in range(len(x)):
            if x[k] < num_imaging_modalities:
                x[k] = 'x' + str(x[k])
            else:
                x[k] = 'y' + str(x[k] - num_imaging_modalities)
        x = ','.join(x)
        return x

    if ',' in left_of_pipe:
        left_of_pipe = sort_one_side(left_of_pipe)

    if not right_of_pipe == '':
        if ',' in right_of_pipe:
            right_of_pipe = sort_one_side(right_of_pipe)

        event = left_of_pipe + '|' + right_of_pipe
    else:
        event = left_of_pipe

    return event


def expand_flag_v2(current_flags, device, is_3d, batch_size=None):
    """
    Make the flag concatenatable with the embedding
    """
    if isinstance(current_flags, list):
        current_flags = torch.tensor(current_flags, dtype=torch.float32, device=device, requires_grad=False)
    current_flags = torch.unsqueeze(current_flags, 0)

    if batch_size is not None:
        current_flags = current_flags.expand(batch_size, -1)  # (batch_size, -1) are the new dims

    current_flags = torch.unsqueeze(current_flags, -1)
    current_flags = torch.unsqueeze(current_flags, -1)

    if is_3d:
        current_flags = torch.unsqueeze(current_flags, -1)

    return current_flags


def create_batch_d_v2(batch, all_modalities, device, json_keys=None, inclusive_mode=True):
    """
    For each subset of modalities, this outputs a dictionary with an 'imaging' tensor and a sequence of 'y0', 'y1', etc
    non-imaging tensors
    """
    modalities_present = batch['modalities_present']
    modalities_imaging = [x for x in all_modalities if 'x' in x]
    # modalities_non_imaging = [x for x in all_modalities if 'y' in x]
    # modality_subsets_imaging = comma_separated_subsets(modalities_imaging)
    # modality_subsets_non_imaging = comma_separated_subsets(modalities_non_imaging)
    modality_subsets_all = comma_separated_subsets(all_modalities)

    """
    Create the (empty) lists
    """
    batch_d = {}
    for subset in modality_subsets_all:
        batch_d[subset] = {'imaging': []}

        for k in range(len(json_keys)):
            batch_d[subset]['y' + str(k)] = []

    """
    Extract the imaging and non-imaging from the batch
    """
    data_imaging = extract_imaging(batch, modalities_imaging)
    data_imaging = np.split(extract_imaging(batch, modalities_imaging), data_imaging.shape[0], 0)

    data_non_imaging = [{} for _ in data_imaging]
    for i, key in enumerate(json_keys):
        current_data = batch[key]

        for j, x in enumerate(current_data):
            data_non_imaging[j]['y' + str(i)] = np.array(x, dtype=np.float32)

    """
    Distribute what is in the batch into the modality keys in the dictionary batch_d
    
    . Iterate over the batch
    . Iterate over all possible modality combos
    """
    for keys_present, x, y in zip(modalities_present, data_imaging, data_non_imaging):

        keys_present_non_imaging = [x for x in keys_present.split(',') if 'y' in x]
        for key in modality_subsets_all:

            # If what is in the current combo *equals* what is present, add the data to the current combo in batch_d
            if key == keys_present:
                # Add the imaging data - in 'non-inclusive mode' imaging is always present at the mo...
                batch_d[key]['imaging'].append(x)

                # Add any non-imaging data
                for modality in keys_present_non_imaging:
                    batch_d[key][modality].append(y[modality])

            # If what is in the current combo *contains* what is present, add the data to the current combo in batch_d
            if inclusive_mode:
                all_keys = key.split(',')
                if sum([a in keys_present for a in all_keys]) == len(all_keys):

                    if 'x' in key:
                        batch_d[key]['imaging'].append(x.clone())

                        # Redact channels that should not be present
                        indices_present = [int(mod.replace('x', '')) for mod in key.split(',') if 'x' in mod]
                        for k in range(len(modalities_imaging)):
                            if k not in indices_present:
                                batch_d[key]['imaging'][-1][:, k, ...] = 0

                    # Add any non-imaging data
                    for modality in keys_present_non_imaging:
                        batch_d[key][modality].append(np.array(y[modality], dtype=np.float32))
                        # MUST REDACT ANY NON-IMAGING HERE THAT IS NOT MENTIONED IN keys, JUST LIKE I DID THE THE IMAGING

    def stack_to_tensor(tensor, squeeze=False):
        tensor = np.stack(tensor, 0)
        if squeeze:
            tensor = np.squeeze(tensor, 1)
        tensor = torch.Tensor(tensor)
        tensor = tensor.to(device, non_blocking=True)
        return tensor

    # Turn each non-empty key into a tensor and each empty key into a None
    for key in modality_subsets_all:
        if len(batch_d[key]['imaging']) > 0:
            batch_d[key]['imaging'] = stack_to_tensor(batch_d[key]['imaging'], squeeze=True)
        else:
            batch_d[key]['imaging'] = None

        for key_non_imaging in batch_d[key]:
            if not key_non_imaging == 'imaging':
                if len(batch_d[key][key_non_imaging]) > 0:
                    batch_d[key][key_non_imaging] = stack_to_tensor(batch_d[key][key_non_imaging])
                else:
                    batch_d[key][key_non_imaging] = None

    return batch_d


# def create_batch_d(batch, imaging_modalities, device, json_keys=None, inclusive_mode=True):
#     """
#     PROBLEM: with >9 modalities: I use string matching like 'x1' in blah, but this will match on 'x1', 'x10', etc
#
#     For each subset of modalities, this outputs an '..._imaging' tensor and a '..._nonimaging' tensor
#     """
#
#     modalities_present = batch['modalities_present']
#     modality_subsets = comma_separated_subsets(imaging_modalities)
#     current_batch = extract_imaging(batch, imaging_modalities)
#     current_batch = np.split(current_batch, current_batch.shape[0], 0)
#
#     batch_d = {}
#
#     # Create empty lists for the json data
#     if json_keys is not None:
#         for k in range(len(json_keys)):
#             # batch_d['y' + str(k)] = []
#             batch_d['y' + str(k)] = {'imaging': [], 'non-imaging': []}
#
#     # Create empty lists for the conjunction of imaging and jsons
#     for subset in modality_subsets:
#         # batch_d[subset] = []
#         batch_d[subset] = {'imaging': [], 'non-imaging': []}
#
#         if json_keys is not None:
#             for k in range(len(json_keys)):
#                 # batch_d[subset + ',y' + str(k)] = []
#                 batch_d[subset + ',y' + str(k)] = {'imaging': [], 'non-imaging': []}
#
#     for current_modalities_present, x in zip(modalities_present, current_batch):
#         for key in modality_subsets:
#             # Iterate over subsets
#             if key == current_modalities_present:
#                 # batch_d[key].append(x)
#                 batch_d[key]['imaging'].append(x)
#
#             elif inclusive_mode:
#                 all_keys = key.split(',')
#                 if sum([a in current_modalities_present for a in all_keys]) == len(all_keys):
#                     # batch_d[key].append(x)
#                     batch_d[key]['imaging'].append(x)
#
#     if json_keys is not None:
#         for k, j_key in enumerate(json_keys):
#             for current_modalities_present, current_json_data in zip(modalities_present, batch[j_key]):
#                 for key in modality_subsets:
#
#                     if key == current_modalities_present:
#                         # batch_d[key + ',y' + str(k)].append(np.array(current_json_data, dtype=np.float32))
#                         batch_d[key + ',y' + str(k)]['non-imaging'].append(np.array(current_json_data, dtype=np.float32))
#
#                     elif inclusive_mode:
#                         all_keys = key.split(',')
#                         if sum([a in current_modalities_present for a in all_keys]) == len(all_keys):
#                             # batch_d[key + ',y' + str(k)].append(np.array(current_json_data, dtype=np.float32))
#                             batch_d[key + ',y' + str(k)]['non-imaging'].append(np.array(current_json_data, dtype=np.float32))
#
#     def stack_to_tensor(tensor, squeeze=True):
#         tensor = np.stack(tensor, 0)
#         if squeeze:
#             tensor = np.squeeze(tensor, 1)
#         tensor = torch.Tensor(tensor)
#         tensor = tensor.to(device, non_blocking=True)
#         return tensor
#
#     for key in modality_subsets:
#         if len(batch_d[key]['imaging']) > 0:
#             batch_d[key]['imaging'] = stack_to_tensor(batch_d[key]['imaging'])
#
#             if json_keys is not None:
#                 for k in range(len(json_keys)):
#                     batch_d[key + ',y' + str(k)]['imaging'] = stack_to_tensor(batch_d[key + ',y' + str(k)]['imaging'])
#                     batch_d[key + ',y' + str(k)]['non-imaging'] = stack_to_tensor(batch_d[key + ',y' + str(k)]['non-imaging'])
#
#     return batch_d


# def create_batch_d(batch, imaging_modalities, device, json_keys=None, inclusive_mode=True):
#     """
#     Problem with >9 modalities: I use string matching like 'x1' in blah, but this will match on 'x1', 'x10', etc
#     """
#     modalities_present = batch['modalities_present']
#     modality_subsets = comma_separated_subsets(imaging_modalities)
#     current_batch = extract_imaging(batch, imaging_modalities)
#     current_batch = np.split(current_batch, current_batch.shape[0], 0)
#
#     batch_d = {}
#     for subset in modality_subsets:
#         batch_d[subset] = []
#
#         if json_keys is not None:
#             for j_key in json_keys:
#                 batch_d[subset + '_' + j_key] = []
#
#     for current_modalities_present, current_batch in zip(modalities_present, current_batch):
#         for key in modality_subsets:
#             # Iterate over subsets
#             if key == current_modalities_present:
#                 batch_d[key].append(current_batch)
#             elif inclusive_mode:
#                 all_keys = key.split(',')
#                 if sum([a in current_modalities_present for a in all_keys]) == len(all_keys):
#                     batch_d[key].append(current_batch)
#
#     if json_keys is not None:
#         for j_key in json_keys:
#             for current_modalities_present, current_json_data in zip(modalities_present, batch[j_key]):
#                 for key in modality_subsets:
#
#                     if key == current_modalities_present:
#                         batch_d[key + '_' + j_key].append(np.array(current_json_data, dtype=np.float32))
#                     elif inclusive_mode:
#                         all_keys = key.split(',')
#                         if sum([a in current_modalities_present for a in all_keys]) == len(all_keys):
#                             batch_d[key + '_' + j_key].append(np.array(current_json_data, dtype=np.float32))
#
#     for key in modality_subsets:
#         if len(batch_d[key]) > 0:
#             batch_d[key] = np.squeeze(np.stack(batch_d[key], 0), 1)
#             batch_d[key] = torch.Tensor(batch_d[key]).to(device, non_blocking=True)
#
#             if json_keys is not None:
#                 for j_key in json_keys:
#                     batch_d[key + '_' + j_key] = np.stack(batch_d[key + '_' + j_key], 0)
#                     batch_d[key + '_' + j_key] = torch.Tensor(batch_d[key + '_' + j_key]).to(device, non_blocking=True)
#
#     return batch_d


def convert_subset_to_one_hot(subset, num_modalities):
    """

    """

    if 'x' in subset:
        subset = re.sub("x", "", subset)

    output = [0] * num_modalities

    if ',' in subset:
        subsets = subset.split(',')

        for s in subsets:
            output[int(s)] = 1

    elif len(subset) > 0:
        output[int(subset)] = 1

    return output


def extract_imaging(batch, modalities):
    output = batch[modalities[0]]
    for k in range(1, len(modalities)):
        output = torch.cat((output, batch[modalities[k]]), 1)

    return output


def print_0_joint_dist(current_distribution):
    dist_string = 'Initialising JOINT: '
    for k, x in enumerate(current_distribution):
        dist_string += 'log p(' + x + ')'
        if k + 1 < len(current_distribution):
            dist_string += ' + '
    print(dist_string)


def make_one_side_of_pipe_numeric(one_side_of_pipe, num_imaging_modalities):
    one_side_of_pipe = one_side_of_pipe.split(',')
    for k in range(len(one_side_of_pipe)):
        if 'x' in one_side_of_pipe[k]:
            one_side_of_pipe[k] = re.sub("x", "", one_side_of_pipe[k])
        if 'y' in one_side_of_pipe[k]:
            one_side_of_pipe[k] = str(int(re.sub("y", "", one_side_of_pipe[k])) + num_imaging_modalities)
    one_side_of_pipe = ','.join(one_side_of_pipe)
    return one_side_of_pipe


def dist_to_vae(event, data, imaging_modalities, non_imaging_modalities):
    """
    Must generalise to N modalities
    :param event:
    :param data:
    :param imaging_modalities:
    :return:
    """
    num_modalities = len(imaging_modalities + non_imaging_modalities)

    if '|' in event:
        left_of_pipe, right_of_pipe = event.split('|')
    else:
        left_of_pipe, right_of_pipe = event, ''

    left_of_pipe = make_one_side_of_pipe_numeric(left_of_pipe, len(imaging_modalities))
    right_of_pipe = make_one_side_of_pipe_numeric(right_of_pipe, len(imaging_modalities))

    """
    Define the flag that can be passed straight to forward_backwards
    """
    flag = [0] * 2 * num_modalities
    flag[int(left_of_pipe)] = 1  # One-hot encoding of left_of_pipe

    if len(right_of_pipe) > 0:
        if ',' in right_of_pipe:
            # Multiple modalities to condition on
            right_of_pipe_split = right_of_pipe.split(',')
            for x in right_of_pipe_split:
                flag[num_modalities + int(x)] = 1
        else:
            # Just one modality to condition on
            flag[num_modalities + int(right_of_pipe)] = 1

    """
    Construct input for prior, P
    Redact anything that is not in right_of_pipe
    """
    if right_of_pipe == '':
        p_input = None
    else:
        p_input = data.clone()
        for k in range(len(imaging_modalities)):
            if str(k) not in right_of_pipe:
                # FUTURE PROBLEMS WITH >9 MODALITIES! This will match 1 in '11', etc!
                p_input[:, k, ...] = 0

    """
    Construct input for posterior, Q
    Redact anything that is not in left_of_pipe or right_of_pipe
    """
    q_input = data.clone()
    for k in range(len(imaging_modalities)):
        if str(k) not in left_of_pipe and str(k) not in right_of_pipe:
            q_input[:, k, ...] = 0

    return flag, p_input, q_input


def expand_flag(flag_as_list, device, data_is_3d):
    out = np.array(flag_as_list)
    out = torch.Tensor(out)
    if data_is_3d:
        out = out.to(device, non_blocking=True).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        out = out.to(device, non_blocking=True).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return out


def joint_distributions(s):
    all_permutations = list(itertools.permutations(s))
    all_permutations = [list(perm) for perm in all_permutations]

    all_joints = []
    for current_permutation in all_permutations:
        current_joint = []

        while len(current_permutation) > 1:
            current_joint.append(current_permutation[0] + '|' + ','.join(current_permutation[1:]))
            current_permutation = current_permutation[1:]

        if len(current_permutation) == 1:
            current_joint.append(current_permutation[0])

        all_joints.append(current_joint)

    if len(all_joints) == 1 and len(all_joints[0]) == 1:
        all_joints = all_joints[0]

    return all_joints


def comma_separated_subsets(s, max_size=None):
    all_subsets = []
    x = len(s)

    for i in range(1, 1 << x):
        subset = [s[j] for j in range(x) if (i & (1 << j))]

        if not (max_size is not None and len(subset) > max_size):
            subset = ','.join(subset)
            all_subsets.append(subset)

    return all_subsets


def pipe_separated_conditionals(s, one_vs_rest_only=True):
    """
    For each proper subset, find the remaining proper subsets, and put them on the other side of a pipe
    """
    conditionals = []
    proper_subsets = comma_separated_subsets(s, max_size=len(s) - 1)

    # Iterate over proper subsets
    for current_subset_left in proper_subsets:

        # Split these subsets into components
        components_of_current_subset_left = current_subset_left.split(',')

        if not (one_vs_rest_only and len(components_of_current_subset_left) > 1):

            # Iterate again over proper subsets
            for proper_subsets_right in proper_subsets:
                # Split these subsets into components
                components_of_proper_subsets_right = proper_subsets_right.split(',')

                # Check there is no overlap in the components
                overlap = list(set(components_of_current_subset_left) & set(components_of_proper_subsets_right))

                if len(overlap) == 0:
                    # If there is no overlap, place one set on the left, the other on the right
                    conditionals.append(current_subset_left + '|' + proper_subsets_right)

    return conditionals


def average_gradients(model):
    """
    Copied from https://pytorch.org/tutorials/intermediate/dist_tuto.html
    This just in-place averages all gradients in 'model' over all ranks in the world
    """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def torch_nansafe_mean(x):
    if x.shape[0] == 0:
        out = torch.mean(torch.zeros(1, device=x.device, requires_grad=False))  # A hack to get correct shape!
    else:
        out = torch.mean(x)
    return out


def flag_randomiser(input):
    """
    Courtesy of https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
    
    I derive all the applicable flags from an example. E.g. if I have a (1, 1) then I can have the flags
    (1, 0), (0, 1) and (1, 1).
    :param input:
    :return:
    """

    # 1) Split long the batch
    N = input.shape[0]
    batch = np.split(input, N, axis=0)
    new_batch = []

    # 2) Iterate over the batch elements
    for x in batch:
        x_shape = np.shape(x)
        new_index = np.zeros_like(x)
        new_index = np.ravel(new_index)

        # 3) Flatten, then find the ones
        locations_of_ones = list(np.flatnonzero(x))

        # 4) Find the unique subsets of ones, then choose one at random
        subsets = []
        for L in range(0, len(locations_of_ones)+1):
            for subset in itertools.combinations(locations_of_ones, L):
                if len(subset) > 0:
                    subsets.append(subset)
        new_locations_of_ones = list(random.choice(subsets))

        # 5) Create the new, replacement index
        for ind in new_locations_of_ones:
            new_index[ind] = 1
        new_index = np.reshape(new_index, x_shape)
        new_batch.append(new_index)

    output = np.squeeze(np.stack(new_batch), 1)

    return output


def count_gradient_nans(gradient_norms, bottom_up_graph_1, top_down_graph, kl, log_likelihood, iteration, hyper_params):
    """
    nan_count_grads, do_not_skip, grad_norm_bu1, grad_norm_td1, grad_norm_td2, grad_norm_td3 = count_gradient_nans(gradient_norms, bottom_up_graph_1, top_down_graph, iteration, hyper_params)
    :param bottom_up_graph_1: 
    :param top_down_graph: 
    :param iteration: 
    :param hyper_params: 
    :return: 
    """
    grad_norm_td3 = None

    c = hyper_params['gradient_clipping_value']
    grad_norm_bu1 = torch.nn.utils.clip_grad_norm_(bottom_up_graph_1.model.parameters(), c).item()
    gradient_norms['bottom_up_graph_1'].append([iteration, grad_norm_bu1])
    grad_norm_td1 = torch.nn.utils.clip_grad_norm_(top_down_graph.latents.parameters(), c).item()
    gradient_norms['top_down_graph_latents'].append([iteration, grad_norm_td1])
    grad_norm_td2 = torch.nn.utils.clip_grad_norm_(top_down_graph.x_mu.parameters(), c).item()
    gradient_norms['top_down_graph_mu'].append([iteration, grad_norm_td2])
    if hyper_params['likelihood'] == 'Gaussian' and \
            hyper_params['separate_output_loc_scale_convs'] and \
            hyper_params['predict_x_var']:
        grad_norm_td3 = torch.nn.utils.clip_grad_norm_(top_down_graph.x_var.parameters(), c).item()
        gradient_norms['top_down_graph_var'].append([iteration, grad_norm_td3])

    if 'gradient_skipping_value' in hyper_params:
        s = hyper_params['gradient_skipping_value']
        do_not_skip = grad_norm_bu1 < s and grad_norm_td1 < s and grad_norm_td2 < s
        if hyper_params['likelihood'] == 'Gaussian' and \
                hyper_params['separate_output_loc_scale_convs'] and \
                hyper_params['predict_x_var']:
            do_not_skip *= grad_norm_td3 < s
    else:
        do_not_skip = True

    # Count infs and NaNs
    nan_count_grads = 0
    nan_count_grads += np.sum(np.isnan(grad_norm_bu1)) + np.sum(np.isinf(grad_norm_bu1))
    nan_count_grads += np.sum(np.isnan(grad_norm_td1)) + np.sum(np.isinf(grad_norm_td1))
    nan_count_grads += np.sum(np.isnan(grad_norm_td2)) + np.sum(np.isinf(grad_norm_td2))
    if hyper_params['likelihood'] == 'Gaussian' and \
            hyper_params['separate_output_loc_scale_convs'] and \
            hyper_params['predict_x_var']:
        nan_count_grads += np.sum(np.isnan(grad_norm_td3)) + np.sum(np.isinf(grad_norm_td3))

    if kl is None:
        nan_count_kl = 0
    else:
        nan_count_kl = torch.isnan(kl).sum() + torch.isinf(kl).sum()
        nan_count_kl = nan_count_kl.item()

    if log_likelihood is None:
        nan_count_loglikelihood = 0
    else:
        nan_count_loglikelihood = torch.isnan(log_likelihood).sum() + torch.isinf(log_likelihood).sum()
        nan_count_loglikelihood = nan_count_loglikelihood.item()

    return nan_count_grads, nan_count_kl, nan_count_loglikelihood, do_not_skip, gradient_norms, grad_norm_bu1, \
           grad_norm_td1, grad_norm_td2


def int_if_not_nan(x):
    if np.sum(np.isnan(x) + np.isinf(x)) == 0:
        return int(x)
    else:
        return x


def gaussian_likelihood(batch_target_features, x_mu, x_var, x_log_var, hyper_params):
    """
    log_likelihood_per_dim, squared_diff_normed = gaussian_likelihood(batch_target_features, x_mu, x_std, x_log_var, hyper_params)
    :param batch_target_features: 
    :param x_mu: 
    :param x_std: 
    :param x_log_var: 
    :param hyper_params: 
    :return: 
    """
    if 'use_abs_not_square' in hyper_params and hyper_params['use_abs_not_square']:
        squared_difference = torch.abs(batch_target_features - x_mu)
    else:
        squared_difference = torch.square(batch_target_features - x_mu)

    if hyper_params['predict_x_var']:
        squared_diff_normed = torch.true_divide(squared_difference, x_var)
        log_likelihood_per_dim = -0.5 * (x_log_var + np.log(2 * np.pi) + squared_diff_normed)
    else:
        if 'kl_multiplier' in hyper_params and not hyper_params['kl_multiplier'] == 1 and not hyper_params['kl_multiplier'] == 0:
            a = hyper_params['kl_multiplier']
            log_likelihood_per_dim = -0.5 * (np.log(a) + np.log(2 * np.pi) +
                                             torch.true_divide(squared_difference, a))
        else:
            log_likelihood_per_dim = -0.5 * (np.log(2 * np.pi) + squared_difference)  # Assuming x_var = I

    return log_likelihood_per_dim, squared_difference

            
def gaussian_output(data_dictionary_x_mu, top_down_graph, hyper_params, num_modalities=1):
    """
    x_mu, x_std, x_var, x_log_var = gaussian_output(data_dictionary_x_mu, top_down_graph, hyper_params, num_modalities=1)
    x_mu, x_std, x_var, x_log_var = gaussian_output(data_dictionary_x_mu, top_down_graph, hyper_params, num_modalities=2)
    :param data_dictionary_x_mu:
    :param top_down_graph:
    :param hyper_params:
    :param num_modalities:
    :return:
    """
    # x_mu = None
    # x_std = None
    # x_var = None
    # x_log_var = None

    if hyper_params['likelihood'] == 'Gaussian' and hyper_params['predict_x_var']:
        # Currently only predicting separate locs and scales for Gaussian p(x|z).
        if hyper_params['separate_output_loc_scale_convs']:
            data_dictionary_x_var = top_down_graph.x_var(data_dictionary_latents)

            if key_is_true(hyper_params, 'predict_x_var_with_sigmoid'):
                lower = hyper_params['variance_output_clamp_bounds'][0]
                upper = hyper_params['variance_output_clamp_bounds'][1]
                x_std = lower + (upper - lower) * torch.sigmoid(data_dictionary_x_var['data'])
                x_var = torch.square(x_std)
                x_log_var = 2 * torch.log(x_std)
            else:
                x_log_var = data_dictionary_x_var['data']
                if hyper_params['variance_output_clamp_bounds'] is not None:
                    x_log_var = torch.clamp(x_log_var,
                                            hyper_params['variance_output_clamp_bounds'][0],
                                            hyper_params['variance_output_clamp_bounds'][1])
                x_var = torch.exp(x_log_var)
                x_std = torch.exp(0.5 * x_log_var)
            x_mu = data_dictionary_x_mu['data']
        else:
            x_mu = data_dictionary_x_mu['data'][:, 0:num_modalities, ...]

            if key_is_true(hyper_params, 'predict_x_var_with_sigmoid'):
                lower = hyper_params['variance_output_clamp_bounds'][0]
                upper = hyper_params['variance_output_clamp_bounds'][1]
                x_std = lower + (upper - lower) * torch.sigmoid(
                    data_dictionary_x_mu['data'][:, num_modalities:2 * num_modalities, ...])
                x_var = torch.square(x_std)
                x_log_var = 2 * torch.log(x_std)
            else:
                x_log_var = data_dictionary_x_mu['data'][:, num_modalities:2 * num_modalities, ...]
                if hyper_params['variance_output_clamp_bounds'] is not None:
                    x_log_var = torch.clamp(x_log_var,
                                            hyper_params['variance_output_clamp_bounds'][0],
                                            hyper_params['variance_output_clamp_bounds'][1])
                x_var = torch.exp(x_log_var)
                x_std = torch.exp(0.5 * x_log_var)
    else:
        x_mu = data_dictionary_x_mu['data']
        x_std = None
        x_var = None
        x_log_var = None

    return x_mu, x_std, x_var, x_log_var


def print_0(hyper_params, x, end=None):
    # Print on lowest rank only
    # devs = [int(d) for d in hyper_params['CUDA_devices']]
    if hyper_params['local_rank'] > 0:  # dist.get_rank() > 0:
        pass
    else:
        # if hyper_params['local_rank'] == np.min(devs):
        if end is not None:
            print(x, end=end, flush=True)
        else:
            print(x)

# def print_on_rank_0(hyper_params, x, end=None):
#     if hyper_params['local_rank'] > 0:  # dist.get_rank() > 0:
#         pass
#     else:
#         if end is not None:
#             print(x, end=end, flush=True)
#         else:
#             print(x)


def tqdm_on_rank_0(hyper_params, x, desc):
    if dist.get_rank() > 0:
        return x
    else:
        return tqdm(x, desc)


class dummy_context_mgr():
    # Copied from https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def zeta(test_case, reference_set, k, DWI_mode=False, index_of_high_signal_in_brain=None):
    # Zeta (slighlty modified version of Tianbo's 18.10.2016 version)

    zeta = np.zeros_like(test_case)
    distances_from_clique = np.abs(test_case[0, :] - reference_set[:, :])

    # Calculate indices, then use that index to rearrange the array
    axis = 0
    ind_pre_calculated = list(np.ix_(*[np.arange(i) for i in distances_from_clique.shape]))
    ind_pre_calculated[axis] = distances_from_clique.argsort(axis)
    val_pre_calculated = distances_from_clique[tuple(ind_pre_calculated)]
    ind_pre_calculated = ind_pre_calculated[axis][0:k, :]
    val_pre_calculated = val_pre_calculated[0:k, :]

    steps = reference_set.shape[1]
    # for j in range(0, steps):
    for j in tqdm(range(0, steps), desc='Computing zeta score'):
        kn = ind_pre_calculated[:, j]
        valn = val_pre_calculated[:, j]

        current_clique = np.reshape(reference_set[kn, j], (k, 1))
        current_feature = test_case[0, j]
        current_clique_mean = np.mean(current_clique)

        if index_of_high_signal_in_brain is not None:
            current_clique_std = np.std(current_clique)
            # number_of_std = 0
            number_of_std = 1
            if current_feature > current_clique_mean + number_of_std * current_clique_std:
                index_of_high_signal_in_brain[0, j] = 1  # Label the high signal regions
            elif current_feature < current_clique_mean - number_of_std * current_clique_std:
                index_of_high_signal_in_brain[0, j] = -1  # Label the low signal regions

        if not (DWI_mode and current_feature <= current_clique_mean):  # For DWIs, we only score the high signal areas
            inter_clique_distances = pdist(current_clique, 'cityblock')
            g = np.mean(valn)  # Mean distance to K nearest neighbours
            c = np.mean(inter_clique_distances)
            zeta[0, j] = g - c
            # s = np.std(inter_clique_distances)
            # if s != 0:
            #     zeta[0, j] /= s  # Not very effective...

        # VIS.printProgress(j + 1, steps, prefix="Computing zeta scores: ", suffix='', decimals=2, barLength=30)

    # sys.stdout.write('\n')
    # sys.stdout.flush()

    ############################################
    # Wherever the inter-clique distances are greater than the mean distance to the clique, record no score
    if index_of_high_signal_in_brain is not None:
        index_of_high_signal_in_brain[zeta < 0] = 0
    # zeta[zeta < 0] = 0  # Ignore scores that come from cliques that are too spread out
    ############################################

    if index_of_high_signal_in_brain is not None:
        return [np.asarray(zeta, dtype=np.float32), index_of_high_signal_in_brain]
    else:
        return np.asarray(zeta, dtype=np.float32)


def make_adc(b_0, b_1000):
    b_0[b_0 == 0] = 1
    output = b_1000 / b_0
    std = np.std(output)
    output[output > 3 * std] = 3 * std
    output[output <= 0] = 1e-6
    output = -np.log(output)
    return output


def create_low_res(data, size):
    original_size = data.size()[2:]
    data = torch.nn.functional.interpolate(input=data, size=size, mode='nearest',
                                           align_corners=None, recompute_scale_factor=True)
    data = torch.nn.functional.interpolate(input=data, size=original_size, mode='nearest',
                                           align_corners=None, recompute_scale_factor=True)
    return data


def filter_by_string(input_list, string):
    idx = [string in s for s in input_list]
    filtered_list = [f for f, i in zip(input_list, idx) if i]
    return filtered_list


def sort_list_and_return_index(l):
    li = []

    for i in range(len(l)):
        li.append([l[i], i])
    li.sort()
    sort_index = []

    for x in li:
        sort_index.append(x[1])
    return sort_index


def custom_compute_grad_norm_(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(parameters, list):
        #list!
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        pass
    else:
        parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device, non_blocking=True) for p in parameters]), norm_type)

    return total_norm


def custom_clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    COPIED FROM PYTORCH WITH CHANGES

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    # if norm_type == inf:
    #     total_norm = max(p.grad.detach().abs().max().to(device, non_blocking=True) for p in parameters)
    # else:
    #     total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device, non_blocking=True) for p in parameters]), norm_type)
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device, non_blocking=True) for p in parameters]), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)

    # if clip_coef < 1:
    #     for p in parameters:
    #         p.grad.detach().mul_(clip_coef.to(p.grad.device))
    # return total_norm

    return clip_coef, total_norm


def kl_perturbed_prior(delta_mu, delta_log_var, log_var):
    """
    This is kl[N(mu+delta_mu, exp(log_var)exp(delta_log_var)) || N(mu, exp(log_var)] 
    """
    vars = torch.exp(log_var)
    delta_vars = torch.exp(delta_log_var)

    squared_delta = torch.square(delta_mu)
    output = 0.5 * (delta_vars + torch.div(squared_delta, vars) - 1 - delta_log_var)

    output = output.view(output.shape[0], -1)
    output = torch.sum(output, dim=1)
    return output


def kl_log_vars(mu, log_var, mu_2=None, log_var_2=None):
    """
    This is kl[N(mu, var) || N(mu_2, var_2)], where var and var_2 are the diagonals of the covariance matrices.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    We parametrise in terms of the log variance to avoid (unstable!) square roots and logarithms. Exp is cool though.
    """
    vars = torch.exp(log_var)

    eps = torch.tensor(1e-5)

    if mu_2 is None:
        # KL between a given (diagonal) Gaussian and N(0,I)
        output = 0.5 * (vars + torch.square(mu) - 1 - log_var)
    else:
        # KL between two given (diagonal) Gaussian
        vars_2 = torch.exp(log_var_2)
        squared_diff = torch.square(mu - mu_2)
        # output = 0.5 * (torch.div(vars + squared_diff, torch.maximum(vars_2, eps)) - 1 + log_var_2 - log_var)
        output = 0.5 * (torch.div(vars + squared_diff, vars_2) - 1 + log_var_2 - log_var)

    output = output.view(output.shape[0], -1)
    output = torch.sum(output, dim=1)
    return output


def kl_vars(mu, var, mu_2=None, var_2=None):
    """
    This is kl[N(mu, var) || N(mu_2, var_2)], where var and var_2 are the diagonals of the covariance matrices.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

    MODIFIED TO TAKE VARS NOT LOG VARS!!
    """

    if mu_2 is None:
        # KL between a given (diagonal) Gaussian and N(0,I)
        output = 0.5 * (var + torch.square(mu) - 1 - torch.log(var))
    else:
        # KL between two given diagonal Gaussians
        squared_diff = torch.square(mu - mu_2)
        output = 0.5 * (torch.div(var + squared_diff, var_2) - 1 + torch.log(var_2) - torch.log(var))

    output = output.view(output.shape[0], -1)
    output = torch.sum(output, dim=1)
    return output


def kl_stds_then_log_vars(mu, std, mu_2=None, log_var_2=None):
    """
    This is kl[N(mu, var) || N(mu_2, var_2)], where var and var_2 are the diagonals of the covariance matrices.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

    MODIFIED TO TAKE STDS THEN LOG VARS!!
    """
    var = torch.square(std)
    if mu_2 is None:
        # KL between a given (diagonal) Gaussian and N(0,I)
        output = 0.5 * (var + torch.square(mu) - 1 - 2 * torch.log(std))
    else:
        # KL between two given (diagonal) Gaussian
        var_2 = torch.exp(log_var_2)
        squared_diff = torch.square(mu - mu_2)
        output = 0.5 * (torch.div(var + squared_diff, var_2) - 1 + log_var_2 - 2 * torch.log(std))

    output = output.view(output.shape[0], -1)
    output = torch.sum(output, dim=1)
    return output


def kl_vars_then_log_vars(mu, var, mu_2=None, log_var_2=None):
    """
    This is kl[N(mu, var) || N(mu_2, var_2)], where var and var_2 are the diagonals of the covariance matrices.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

    MODIFIED TO TAKE VARS THEN LOG VARS!!
    """

    if mu_2 is None:
        # KL between a given (diagonal) Gaussian and N(0,I)
        output = 0.5 * (var + torch.square(mu) - 1 - torch.log(var))
    else:
        # KL between two given (diagonal) Gaussian
        var_2 = torch.exp(log_var_2)
        squared_diff = torch.square(mu - mu_2)
        output = 0.5 * (torch.div(var + squared_diff, var_2) - 1 + log_var_2 - torch.log(var))

    output = output.view(output.shape[0], -1)
    output = torch.sum(output, dim=1)
    return output


def kl_stds(mu, std, mu_2=None, std_2=None, kl_mask=None):
    """
    This is kl[N(mu, var) || N(mu_2, var_2)], where var and var_2 are the diagonals of the covariance matrices.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

    MODIFIED TO TAKE STDs NOT LOG VARS!!
    """
    eps = torch.tensor(1e-5)

    var = torch.square(std)
    if mu_2 is None:
        # KL between a given (diagonal) Gaussian and N(0,I)
        output = 0.5 * (var + torch.square(mu) - 1 - 2 * torch.log(std))
    else:
        # KL between two given (diagonal) Gaussian
        var_2 = torch.square(std_2)
        squared_diff = torch.square(mu - mu_2)
        output = 0.5 * (torch.div(var + squared_diff, torch.maximum(var_2, eps)) - 1 + 2 * torch.log(std_2) - 2 * torch.log(std))

    # output = torch.clamp(output, 0, 1e5)

    if kl_mask is not None:
        output = output * kl_mask

    output = output.view(output.shape[0], -1)
    output = torch.sum(output, dim=1)
    return output


def iterate_through_h5py(open_h5py_file):
    variables = open_h5py_file.items()
    num_vars = 0
    names = []
    shapes = []
    for var in variables:
        num_vars += 1
        name = var[0]
        data = var[1]
        # print("Name " + name)  # Name
        if type(data) is h5py.Dataset:
            # If DataSet pull the associated Data
            # If not a dataset, you may need to access the element sub-items
            shape = data.shape
            # print("Value" + str(shape))  # NumPy Array / Value
            names.append(name)
            shapes.append(shape)
    return num_vars, names, shapes


def key_is_true(dict, key):
    """
    If a key exists AND is True, return True, otherwise return False.
    """
    return key in dict and dict[key]


def non_batch_dims(tensor):
    """
    Return a tuple of all the dims in a PyTorch tensor except the first.
    Useful for reducing over non-batch dimensions
    """
    return tuple(range(1, len(tensor.shape)))


def compute_cnn_output_shape(data_shape, pooling_per_layer, channels, is_colour):
    """
    Given the input arguments, compute the output shape & dimensionality
    """
    current_shape = data_shape
    if is_colour:
        # Remove the colour channel from the input
        current_shape = current_shape[1::]
    for pooling_amount in pooling_per_layer:
        current_shape = [d / pooling_amount for d in current_shape]
    final_shape = [channels[-1]] + current_shape  # Using torch's b, c, h, w, d convention
    final_shape = [int(d) for d in final_shape]
    final_dimensionality = np.prod(final_shape)
    return final_shape, final_dimensionality


def sum_weight_l2_norms(parameters, multiplier=None):
    """
    Given parameters=model.parameters() where model is a PyTorch model, this iterates through the list and tallies
    the L2 norms of all the weight matrices.
    """
    l2_reg = 0
    for param in parameters:
        if len(list(param.size())) == 2:
            l2_reg += torch.mean(torch.square(param))

    if multiplier is not None:
        l2_reg = multiplier * l2_reg
    return l2_reg


def remove_list_elements(x, a, b, c):
    for s in x:
        if not (s in a and s in b and s in c):
            print("Subject present in only a proper subset of the modalities: " + s)
            x.remove(s)
    return x


def sum_non_bias_l2_norms(parameters, multiplier=None):
    """
    Given parameters=model.parameters() where model is a PyTorch model, this iterates through the list and tallies
    the L2 norms of all the non-bias tensors.
    """
    l2_reg = 0
    for param in parameters:
        if len(list(param.size())) > 1:
            l2_reg += torch.mean(torch.square(param))

    if multiplier is not None:
        l2_reg = multiplier * l2_reg
    return l2_reg


def sum_non_bias_l1_norms(parameters, multiplier=None):
    """
    Given parameters=model.parameters() where model is a PyTorch model, this iterates through the list and tallies
    the L1 norms of all the non-bias matrices.
    """
    l2_reg = 0
    for param in parameters:
        if len(list(param.size())) > 1:
            l2_reg += torch.mean(torch.abs(param))

    if multiplier is not None:
        l2_reg = multiplier * l2_reg
    return l2_reg


def multi_gpu(module, input, device_ids, output_device=None):
    """
    This is copied verbatim from https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html.
    As long as you use dictionaries rather than lists whenever you pass more than one input to a 'forward' function,
    this makes data parallelisation trivial.
    """
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    output = nn.parallel.gather(outputs, output_device)

    return output


def np_safe_divide(numer, denom):
    denom[denom < 1e-5] = 1
    return numer / denom


def count_unique_parameters(parameters):
    # Only counts unique params
    count = 0
    list_of_names = []
    for p in parameters:
        name = p[0]
        param = p[1]
        if name not in list_of_names:
            list_of_names.append(name)
            count += np.prod(param.size())
    return count


def count_parameters(parameters):
    count = 0
    for p in parameters:
        count += np.prod(p.size())
    return count


def bn_diagnosis(model, start):
    print(start)
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm1d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm3d):
            print("EMA_updating: " + str(mod.training))
            print("Weights_updating: " + str(mod.weight.requires_grad))
            print("Biases_updating: " + str(mod.bias.requires_grad))
            print("_________________________")


def extract_non_batch_norm_params(model_list):
    """
    Extract and add to a list all the parameters in each model in model_list that do not contain '*batch_norm*'
    in their name.
    """
    params_less_bn = []
    for model in model_list:
        for name, param in model.named_parameters():
            if 'batch_norm' not in name:
                params_less_bn.append(param)
    return params_less_bn


def freeze_batch_norm(model_list, verbose):
    """"
    Put any BatchNormNd layers into eval() mode, AND stop their weights and biases receiving gradients.
    """
    was_training = False
    was_receiving_grads = False
    for model in model_list:
        count = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or \
                    isinstance(module, nn.BatchNorm3d):
                count += 1

                if module.training:
                    was_training = True

                module.eval()

                if hasattr(module.weight, 'requires_grad'):
                    # This batch norm module contains affine parameters
                    if module.weight.requires_grad or module.bias.requires_grad:
                        was_receiving_grads = True

                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
        if verbose:
            print("Frozen " + str(count) + " batch norm layers!")
    return was_training, was_receiving_grads


def loop_through_module_parameters_and_print_sizes(module):
    for param in module.parameters():
        print(type(param.data), param.size())
    return


def extract_linear_layer_weights(modules):
    weight_matrices = []
    for module in modules:
        if isinstance(module, nn.Linear):
            weight_matrices.append(module.weight)
    return weight_matrices
