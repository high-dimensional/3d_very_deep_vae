import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist


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


def count_gradient_nans(gradient_norms, bottom_up_graph_1, top_down_graph, kl, log_likelihood, iteration, hyper_params):
    """
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
    if hyper_params['local_rank'] > 0:  # dist.get_rank() > 0:
        pass
    else:
        # if hyper_params['local_rank'] == np.min(devs):
        if end is not None:
            print(x, end=end, flush=True)
        else:
            print(x)


def tqdm_on_rank_0(hyper_params, x, desc):
    if hyper_params['local_rank'] > 0:
        return x
    else:
        return tqdm(x, desc)


class dummy_context_mgr():
    # Copied from https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


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
