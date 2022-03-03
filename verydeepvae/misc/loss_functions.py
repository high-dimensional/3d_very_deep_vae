import torch
import numpy as np
import matplotlib as mpl
mpl.use("AGG")
import misc


"""
Custom loss functions are defined here.
"""


def logistic_pdf(x, mu, s):
    '''
    V1: Naive implementation for starters!
    '''
    numerator = torch.exp(-torch.true_divide(x - mu, s))
    denominator = s * torch.square(1 + numerator)
    pdf = torch.trunc(numerator, denominator)
    return pdf


def sample_from_mixture_of_logistics(pi, mu, s, M):
    '''
    V1: Naive implementation for starters!
    '''

    shape_mu_orig = list(mu.shape)
    shape_mu_final = shape_mu_orig[:]
    shape_mu_final[1] = int(shape_mu_final[1] / M)

    mu = torch.reshape(mu, [mu.shape[0], -1, 32, 32, M])
    pi = torch.reshape(pi, [mu.shape[0], -1, 32, 32, M])
    s = torch.reshape(s, [mu.shape[0], -1, 32, 32, M])

    max = 255.

    mu = torch.sigmoid(mu) * max
    pi_max = torch.argmax(pi, dim=-1, keepdim=False)  # Choose the mixture component for each pixel
    s = torch.nn.functional.softplus(s, beta=1, threshold=20) + 1e-8

    # Logistic distribution
    base_distribution = torch.distributions.Uniform(0, 1)
    transforms = [torch.distributions.transforms.SigmoidTransform().inv,
                  torch.distributions.transforms.AffineTransform(loc=mu, scale=s)]
    logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)
    sample = logistic.sample()  # FMI, '.sample()' has a torch.no_grad, '.rsample()' doesn't

    sample = sample.cpu().detach().numpy()
    pi_max = pi_max.cpu().detach().numpy()

    sample = np.transpose(sample, [4, 0, 1, 2, 3])
    sample = np.reshape(sample, [M, -1])
    pi_max = np.reshape(pi_max, [-1])

    sample = np.hsplit(sample, sample.shape[1])
    sample = [q[r, 0] for q, r in zip(sample, pi_max)]
    sample = np.stack(sample)
    sample = np.reshape(sample, shape_mu_final)

    sample = np.round(sample)

    return sample


def mixture_of_logistics(x, pi, mu, s, M):
    '''
    V3: Still naive...
    '''

    mu = torch.reshape(mu, list(x.shape) + [M])  # B x C x H x W
    pi = torch.reshape(pi, list(x.shape) + [M])
    s = torch.reshape(s, list(x.shape) + [M])

    w = 0.5
    min = 0
    max = 255

    mu = torch.sigmoid(mu) * max
    pi = torch.nn.functional.softmax(pi, dim=-1)
    s = torch.nn.functional.softplus(s, beta=1, threshold=20) + 1e-8

    x = x.unsqueeze(-1)  # B x C x H x W x 1, trailing singleton is for broadcasting with (trailing) mixture comp dim

    below = torch.true_divide(x - mu - w, s)
    above = torch.true_divide(x - mu + w, s)

    case_low = torch.sigmoid(above)
    case_medium = torch.sigmoid(above) - torch.sigmoid(below)
    case_high = 1 - torch.sigmoid(below)

    mask_low = x <= min
    mask_high = x >= max
    mask_low = mask_low.type(torch.float32)
    mask_high = mask_high.type(torch.float32)
    mask_medium = (1-mask_low) * (1-mask_high)

    case_low = case_low * mask_low  # Problem with this is that it will propogate any NaNs lurking in case_low...
    case_medium = case_medium * mask_medium
    case_high = case_high * mask_high

    logistics = case_low + case_medium + case_high
    mixture = pi * logistics
    likelihood = torch.clamp(torch.sum(mixture, -1), 1e-8, 1e8)
    log_likelihood = torch.log(likelihood)

    return log_likelihood


# def mixture_of_logistics(x, pi, mu, s, M):
#     '''
#     V2: Less naive implementation for starters!
#     '''
#
#     # x_shape = x.shape
#     # new_shape = x_shape[:]
#     # new_shape[1] = new_shape[1] / M
#     # new_shape = [x_shape[0], int(x_shape[1]/M)] + x_shape[]
#
#     mu = torch.reshape(mu, list(x.shape) + [M])  # B x C x H x W
#     pi = torch.reshape(pi, list(x.shape) + [M])
#     s = torch.reshape(s, list(x.shape) + [M])
#
#     pi = torch.nn.functional.softmax(pi, dim=-1)
#     s = torch.nn.functional.softplus(s, beta=1, threshold=20) + 1e-8
#
#     x = x.unsqueeze(-1)  # B x C x H x W x 1, trailing singleton is for broadcasting with (trailing) mixture comp dim
#
#     w = 0.5 / 255.
#     min = 0
#     max = 255 / 255.
#
#     a = torch.true_divide(x - mu + w, s)
#     b = torch.true_divide(x - mu - w, s)
#     case_low = a - torch.nn.functional.softplus(a)
#     case_medium = torch.log(torch.clamp(torch.sigmoid(a) - torch.sigmoid(b), 1e-8, 1e8))
#     case_high = -torch.nn.functional.softplus(b)
#
#     mask_low = x <= min
#     mask_high = x >= max
#     mask_low = mask_low.type(torch.float32)
#     mask_high = mask_high.type(torch.float32)
#     mask_medium = (1-mask_low) * (1-mask_high)
#
#     case_low = case_low * mask_low  # Problem with this is that it will propogate any NaNs lurking in case_low...
#     case_medium = case_medium * mask_medium
#     case_high = case_high * mask_high
#
#     log_logistics = case_low + case_medium + case_high
#     mixture = pi * log_logistics
#     log_likelihood = torch.sum(mixture, -1)
#     log_likelihood = -torch.log(log_likelihood)
#
#     return log_likelihood


# def mixture_of_logistics(x, pi, mu, s, M):
#     '''
#     V1: Naive implementation for starters!
#     '''
#
#     x = x.unsqueeze(-1)  # B x C x H x W x 1, trailing singleton is for broadcasting with (trailing) mixture comp dim
#
#     mu = torch.reshape(mu, [mu.shape[0], 1, 32, 32, M])
#
#     pi = torch.reshape(pi, [mu.shape[0], 1, 32, 32, M])
#     pi = torch.nn.functional.softmax(pi, dim=-1)
#
#     s = torch.reshape(s, [mu.shape[0], 1, 32, 32, M])
#     s = torch.nn.functional.softplus(s, beta=1, threshold=20) + 1e-12
#
#     width = 0.5
#     min = 0
#     max = 255
#     pos_infinity = 1e8
#     neg_infinity = -1e8
#
#     arg_1 = x + width
#     arg_1[x == max] = pos_infinity
#     arg_1 = torch.sigmoid(torch.true_divide(arg_1 - mu, s))
#
#     arg_2 = x - width
#     arg_2[x == min] = neg_infinity
#     arg_2 = torch.sigmoid(torch.true_divide(arg_2 - mu, s))
#
#     combined = pi * (arg_1 - arg_2)
#     pdf = torch.sum(combined, -1)
#     pdf = torch.clamp(pdf, 1e-12, 1e8)
#     log_likelihood = torch.log(pdf)
#
#     return log_likelihood


def ae_with_split_classifier_v2(x, recon, embedding, predicted_trailing_dim, mask, apply_mask=False, veto_mask_update=False):
    """
    """
    squared_difference = torch.square(x - recon)

    if apply_mask:
        if not veto_mask_update:
            mask_new = torch.std(x, dim=(0, 1), keepdim=True)
            mask_new[mask_new > 0] = 1

            if mask is None:
                mask = mask_new
            else:
                mask = torch.bitwise_and(mask.to(torch.int8), mask_new.to(torch.int8)).to(torch.float32)

        squared_difference = torch.mul(squared_difference, mask)

    with torch.no_grad():
        true_trailing_dim = torch.reshape(embedding[:, -1], [embedding.shape[0], 1])
    squared_diff = torch.square(true_trailing_dim - predicted_trailing_dim)
    abs_diff = torch.abs(true_trailing_dim - predicted_trailing_dim)
    trailing_dim_mse = torch.mean(squared_diff, [1])  # Per datum

    mean_se = torch.mean(squared_difference, [1, 2, 3])  # Per datum
    objective = torch.mean(mean_se + trailing_dim_mse)

    average_accuracy = torch.mean(torch.mean(abs_diff, [1]))
    average_mse = torch.mean(mean_se)

    return objective, average_accuracy, average_mse, mask


def ae_with_split_classifier(x, recon, embedding, logits, labels, mask, apply_mask=False, veto_mask_update=False):
    """
    """
    squared_difference = torch.square(x - recon)

    if apply_mask:
        if not veto_mask_update:
            mask_new = torch.std(x, dim=(0, 1), keepdim=True)
            mask_new[mask_new > 0] = 1

            if mask is None:
                mask = mask_new
            else:
                mask = torch.bitwise_and(mask.to(torch.int8), mask_new.to(torch.int8)).to(torch.float32)

        squared_difference = torch.mul(squared_difference, mask)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='none')
    probabilities = torch.sigmoid(logits)
    predictions = probabilities > 0.5
    predictions = predictions.float()

    mean_se = torch.mean(squared_difference, [1, 2, 3])
    mean_bce = torch.mean(bce, [1])
    objective = torch.mean(mean_se + mean_bce)

    average_accuracy = torch.mean(predictions)
    average_mse = torch.mean(mean_se)

    return objective, average_accuracy, average_mse, mask


def det_gen_ae(x, recon, embedding, mask, entropy_coeff=1, apply_mask=False, veto_mask_update=False):
    """
    """
    squared_difference = torch.square(x - recon)

    if apply_mask:
        if not veto_mask_update:
            mask_new = torch.std(x, dim=(0, 1), keepdim=True)
            mask_new[mask_new > 0] = 1

            if mask is None:
                mask = mask_new
            else:
                mask = torch.bitwise_and(mask.to(torch.int8), mask_new.to(torch.int8)).to(torch.float32)

        squared_difference = torch.mul(squared_difference, mask)

    logits = embedding
    probs = torch.sigmoid(embedding)
    entropy = -torch.mul(probs, logits)
    entropy = torch.sum(entropy, 1)

    mse = torch.mean(squared_difference, [1, 2, 3])
    objective = torch.mean(mse - entropy_coeff * entropy)

    average_entropy = torch.mean(entropy)
    average_mse = torch.mean(mse)

    return objective, average_entropy, average_mse, mask


def kl_divergence(mu, log_var):
    """
    kl, https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    """
    output = 0.5 * (torch.exp(log_var) + torch.square(mu) - 1 - log_var)
    output = torch.sum(output, dim=1)
    return output


def vlb(x, x_mu, x_log_var, z_mu, z_log_var, mask, apply_mask=False, veto_mask_update=False, gaussian_likelihood=True,
        kl_multiplier=1):
    """
    vlb
    """
    squared_difference = torch.square(x - x_mu)

    if gaussian_likelihood:
        if x_log_var is None:
            log_likelihood_per_dim = -0.5 * squared_difference  # No np.log(2 * np.pi) to make it easier to interpret
        else:
            x_var = torch.exp(x_log_var)  # x_log_var has been clamped, so this is usable in a numerator & denominator
            log_likelihood_per_dim = -0.5 * (x_log_var + np.log(2 * np.pi) + torch.true_divide(squared_difference, x_var))
    else:
        log_likelihood_per_dim = \
            -torch.nn.functional.binary_cross_entropy_with_logits(input=x_mu.reshape(x_mu.shape[0], -1),
                                                                  target=x.reshape(x.shape[0], -1),
                                                                  reduction='none')
        log_likelihood_per_dim = log_likelihood_per_dim.reshape(x.shape)

    if apply_mask:
        if not veto_mask_update:
            mask_new = torch.std(x, dim=(0, 1), keepdim=True)
            mask_new[mask_new > 0] = 1

            if mask is None:
                mask = mask_new
            else:
                mask = torch.bitwise_and(mask.to(torch.int8), mask_new.to(torch.int8)).to(torch.float32)

        log_likelihood_per_dim = torch.mul(log_likelihood_per_dim, mask)
        squared_difference = torch.mul(squared_difference, mask)

    log_likelihood = torch.sum(log_likelihood_per_dim, dim=misc.non_batch_dims(log_likelihood_per_dim))

    kl = kl_divergence(z_mu, z_log_var)
    vlb = log_likelihood - kl_multiplier * kl

    average_vlb = torch.mean(vlb)
    average_kl = torch.mean(kl)
    average_se = torch.mean(squared_difference)

    return -average_vlb, average_kl, average_se, mask


def vdeep_vlb(x, generative_dictionary, mask, apply_mask=False, veto_mask_update=False, gaussian_likelihood=True,
        kl_multiplier=1):
    """
    vlb
    """
    kl = recognition_dictionary['KL_list']
    kl = torch.stack(kl)
    kl = torch.sum(kl, 0)

    x_mu = generative_dictionary['data']
    squared_difference = torch.square(batch_features - x_mu)
    log_likelihood_per_dim = -0.5 * squared_difference
    log_likelihood = torch.sum(log_likelihood_per_dim, dim=misc.non_batch_dims(log_likelihood_per_dim))

    vlb = log_likelihood - kl

    loss = torch.mean(-vlb)
    kl = torch.mean(kl)
    mse = torch.mean(squared_difference)


    squared_difference = torch.square(x - x_mu)

    if gaussian_likelihood:
        if x_log_var is None:
            log_likelihood_per_dim = -0.5 * squared_difference  # No np.log(2 * np.pi) to make it easier to interpret
        else:
            x_var = torch.exp(x_log_var)  # x_log_var has been clamped, so this is usable in a numerator & denominator
            log_likelihood_per_dim = -0.5 * (x_log_var + np.log(2 * np.pi) + torch.true_divide(squared_difference, x_var))
    else:
        log_likelihood_per_dim = \
            -torch.nn.functional.binary_cross_entropy_with_logits(input=x_mu.reshape(x_mu.shape[0], -1),
                                                                  target=x.reshape(x.shape[0], -1),
                                                                  reduction='none')
        log_likelihood_per_dim = log_likelihood_per_dim.reshape(x.shape)

    if apply_mask:
        if not veto_mask_update:
            mask_new = torch.std(x, dim=(0, 1), keepdim=True)
            mask_new[mask_new > 0] = 1

            if mask is None:
                mask = mask_new
            else:
                mask = torch.bitwise_and(mask.to(torch.int8), mask_new.to(torch.int8)).to(torch.float32)

        log_likelihood_per_dim = torch.mul(log_likelihood_per_dim, mask)
        squared_difference = torch.mul(squared_difference, mask)

    log_likelihood = torch.sum(log_likelihood_per_dim, dim=misc.non_batch_dims(log_likelihood_per_dim))

    kl = kl_divergence(z_mu, z_log_var)
    vlb = log_likelihood - kl_multiplier * kl

    average_vlb = torch.mean(vlb)
    average_kl = torch.mean(kl)
    average_se = torch.mean(squared_difference)

    return -average_vlb, average_kl, average_se, mask
