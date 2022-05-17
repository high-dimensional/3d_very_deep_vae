import torch
import torch.cuda.amp as amp
import numpy as np
from ..misc import misc
import torch.distributed as dist


def go(input_dict):
    bottom_up_graph_1 = input_dict["bottom_up_graph_1"]
    top_down_graph = input_dict["top_down_graph"]
    hyper_params = input_dict["hyper_params"]
    optimizer = input_dict["optimizer"]
    device = input_dict["device"]
    validation_mask = input_dict["validation_mask"]
    params = input_dict["params"]
    scaler = input_dict["scaler"]
    epoch = input_dict["epoch"]
    writer = input_dict["writer"]
    training = input_dict["training"]
    progress_bar_text = input_dict["progress_bar_text"]
    summary_text_prefix = input_dict["summary_text_prefix"]
    writer_prefix = input_dict["writer_prefix"]
    loader = input_dict["loader"]
    batch_count = len(loader)

    if training:
        bottom_up_graph_1.model.train()
        top_down_graph.latents.train()
        top_down_graph.x_mu.train()
        if (
            hyper_params["likelihood"] == "Gaussian"
            and hyper_params["separate_output_loc_scale_convs"]
            and hyper_params["predict_x_var"]
        ):
            top_down_graph.x_var.train()
        progress_bar_prefix = "Epoch " + str(epoch) + ") "
    else:
        bottom_up_graph_1.model.eval()
        top_down_graph.latents.eval()
        top_down_graph.x_mu.eval()
        if (
            hyper_params["likelihood"] == "Gaussian"
            and hyper_params["separate_output_loc_scale_convs"]
            and hyper_params["predict_x_var"]
        ):
            top_down_graph.x_var.eval()
        progress_bar_prefix = ""

    loss_tally = 0
    kl_tally = 0
    mse_tally = 0
    elbo_tally = 0
    nll_tally = 0
    iteration = 0
    skipped_updates = 0
    std_mean_tally = 0
    std_std_tally = 0
    skipped_updates_nan = 0
    skipped_updates_too_big = 0
    kl_all_tallies = {}

    gradient_norms = {
        "bottom_up_graph_1": [],
        "top_down_graph_latents": [],
        "top_down_graph_mu": [],
    }
    if (
        hyper_params["likelihood"] == "Gaussian"
        and hyper_params["separate_output_loc_scale_convs"]
        and hyper_params["predict_x_var"]
    ):
        gradient_norms["top_down_graph_var"] = []

    grad_norm_bu1 = 0
    grad_norm_td1 = 0
    grad_norm_td2 = 0

    tqdm_obj = misc.tqdm_on_rank_0(
        hyper_params, loader, desc=progress_bar_prefix + progress_bar_text
    )

    for batch in tqdm_obj:
        optimizer.zero_grad()
        iteration += 1

        with amp.autocast(hyper_params["half_precision"]):

            if hyper_params["use_nii_data"]:
                current_input = batch["full_brain"].to(
                    device
                )  # In this case use dictionaries
            else:
                current_input = batch[0].to(device)

            input_dictionary_1 = {"data": current_input}
            batch_target_features = current_input
            data_dictionary_1 = bottom_up_graph_1.model(input_dictionary_1)
            data_dictionary = {"data": data_dictionary_1["data"], "KL_list": []}
            for key in data_dictionary_1:
                data_dictionary["encoder1_" + key] = data_dictionary_1[key]

            data_dictionary_latents = top_down_graph.latents(data_dictionary)
            data_dictionary_x_mu = top_down_graph.x_mu(data_dictionary_latents)

            if hyper_params["likelihood"] == "Gaussian":
                x_mu, x_std, x_var, x_log_var = misc.gaussian_output(
                    data_dictionary_x_mu, top_down_graph, hyper_params, num_modalities=1
                )
            else:
                logits = data_dictionary_x_mu["data"]

            if hyper_params["likelihood"] == "Gaussian":
                log_likelihood_per_dim, squared_difference = misc.gaussian_likelihood(
                    batch_target_features, x_mu, x_var, x_log_var, hyper_params
                )

            else:
                log_likelihood_per_dim = -torch.nn.functional.binary_cross_entropy_with_logits(
                    input=logits.reshape(logits.shape[0], -1),
                    target=batch_target_features.reshape(
                        batch_target_features.shape[0], -1
                    ),
                    reduction="none",
                )
                squared_difference = torch.mean(-log_likelihood_per_dim)
                log_likelihood_per_dim = log_likelihood_per_dim.reshape(
                    batch_target_features.shape
                )

            if validation_mask is not None:
                log_likelihood_per_dim = torch.mul(
                    log_likelihood_per_dim, validation_mask
                )
                squared_difference = torch.mul(squared_difference, validation_mask)

            log_likelihood = torch.sum(
                log_likelihood_per_dim, dim=misc.non_batch_dims(log_likelihood_per_dim)
            )

            kl_all = data_dictionary_latents["KL_list"]

            if "KLs_to_use" in hyper_params and hyper_params["KLs_to_use"]:
                for i, j in enumerate(hyper_params["KLs_to_use"]):
                    if not j:
                        # kl_all[-1-i] = torch.zeros_like(kl_all[-1-i])
                        kl_all[-1 - i] *= 1e-10

            kl = torch.stack(kl_all)
            kl_adjusted = torch.stack(kl_all)

            kl_for_loss = torch.sum(kl_adjusted, 0)
            kl = torch.sum(kl, 0)
            kl_all = [torch.mean(a) for a in kl_all]
            vlb = log_likelihood - kl_for_loss

            loss = torch.mean(-vlb)
            kl = torch.mean(kl)
            mse = torch.mean(squared_difference)
            elbo_average = torch.mean(-vlb)
            nll_average = torch.mean(-log_likelihood)

            loss += misc.sum_non_bias_l2_norms(params, hyper_params["l2_reg_coeff"])

            if (
                hyper_params["likelihood"] == "Gaussian"
                and hyper_params["predict_x_var"]
            ):
                if (
                    "x_std_l2_penalty" in hyper_params
                    and hyper_params["x_std_l2_penalty"] > 0
                ):
                    loss += hyper_params["x_std_l2_penalty"] * torch.sum(
                        torch.square(x_std - torch.ones_like(x_std))
                    )

            if (
                hyper_params["likelihood"] == "Gaussian"
                and hyper_params["predict_x_var"]
            ):
                std_mean = torch.mean(x_std)
                std_std = torch.std(x_std)

        if training:
            nan_count = torch.isnan(loss).sum() + torch.isinf(loss).sum()
            nan_count = nan_count.item()

            if nan_count == 0:
                scaler.scale(loss).backward()

                if "gradient_clipping_value" in hyper_params:
                    scaler.unscale_(optimizer)

                    (
                        nan_count_grads,
                        nan_count_kl,
                        nan_count_loglikelihood,
                        do_not_skip,
                        gradient_norms,
                        grad_norm_bu1,
                        grad_norm_td1,
                        grad_norm_td2,
                    ) = misc.count_gradient_nans(
                        gradient_norms,
                        bottom_up_graph_1,
                        top_down_graph,
                        kl,
                        log_likelihood,
                        iteration,
                        hyper_params,
                    )

                    if (
                        epoch == 1 and iteration < hyper_params["warmup_iterations"]
                    ) or (
                        do_not_skip
                        and nan_count_kl
                        == nan_count_loglikelihood
                        == nan_count_grads
                        == 0
                    ):
                        scaler.step(optimizer)
                    else:
                        if (
                            nan_count_kl > 0
                            or nan_count_loglikelihood > 0
                            or not nan_count_grads > 0
                        ):
                            skipped_updates_nan += 1

                        if not do_not_skip:
                            skipped_updates_too_big += 1

                        if (
                            nan_count_kl > 0
                            or nan_count_loglikelihood > 0
                            or not nan_count_grads > 0
                            or not do_not_skip
                        ):
                            skipped_updates += 1

                else:
                    # No gradient clipping in this case
                    scaler.step(optimizer)

                scaler.update()
            else:
                misc.print_0(hyper_params, "Detected NaN: skipping update")

            # In train model we let each DDP process do its own thing (apart from synchronising gradients)
            loss_tally += loss.item()
            kl_tally += kl.item()
            elbo_tally += elbo_average.item()
            nll_tally += nll_average.item()

            if (
                hyper_params["likelihood"] == "Gaussian"
                and hyper_params["predict_x_var"]
            ):
                std_mean_tally += std_mean.item()
                std_std_tally += std_std.item()

            # Tally the individual KLs
            for i in range(len(kl_all)):
                current_key = str(i)
                kl_current = kl_all[i].item()
                if current_key in kl_all_tallies:
                    kl_all_tallies[current_key] += kl_current
                else:
                    kl_all_tallies[current_key] = kl_current

            if hyper_params["local_rank"] == 0:
                tqdm_obj_dict = {
                    "MSE": mse.cpu().detach().numpy(),
                    "KL": kl.cpu().detach().numpy(),
                    "SkipNaN": skipped_updates_nan,
                    "SkipBig": skipped_updates_too_big,
                }

                if "gradient_clipping_value" in hyper_params:
                    tqdm_obj_dict["BU1"] = misc.int_if_not_nan(grad_norm_bu1)
                    tqdm_obj_dict["TD1"] = misc.int_if_not_nan(grad_norm_td1)
                    tqdm_obj_dict["TD2"] = misc.int_if_not_nan(grad_norm_td2)

                tqdm_obj.set_postfix(tqdm_obj_dict)

        else:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(kl, op=dist.ReduceOp.SUM)
            dist.all_reduce(elbo_average, op=dist.ReduceOp.SUM)
            dist.all_reduce(nll_average, op=dist.ReduceOp.SUM)
            if (
                hyper_params["likelihood"] == "Gaussian"
                and hyper_params["predict_x_var"]
            ):
                dist.all_reduce(std_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(std_std, op=dist.ReduceOp.SUM)

            # Divide by number of DDP processes
            loss /= hyper_params["global_world_size"]
            kl /= hyper_params["global_world_size"]
            elbo_average /= hyper_params["global_world_size"]
            nll_average /= hyper_params["global_world_size"]
            if (
                hyper_params["likelihood"] == "Gaussian"
                and hyper_params["predict_x_var"]
            ):
                std_mean /= hyper_params["global_world_size"]
                std_std /= hyper_params["global_world_size"]

            loss_tally += loss.item()
            kl_tally += kl.item()
            elbo_tally += elbo_average.item()
            nll_tally += nll_average.item()
            if (
                hyper_params["likelihood"] == "Gaussian"
                and hyper_params["predict_x_var"]
            ):
                std_mean_tally += std_mean.item()
                std_std_tally += std_std.item()

            # Tally the individual KLs
            for i in range(len(kl_all)):
                dist.reduce(kl_all[i], 0, op=dist.ReduceOp.SUM)
                kl_all[i] /= hyper_params["global_world_size"]

            # Tally the individual KLs
            for i in range(len(kl_all)):
                current_key = str(i)
                kl_current = kl_all[i].item()
                if current_key in kl_all_tallies:
                    kl_all_tallies[current_key] += kl_current
                else:
                    kl_all_tallies[current_key] = kl_current

            if hyper_params["local_rank"] == 0:
                tqdm_obj.set_postfix(
                    {"MSE": mse.cpu().detach().numpy(), "KL": kl.cpu().detach().numpy()}
                )

    # The tallies are all sums of batch-wise errors
    loss_tally /= batch_count
    kl_tally /= batch_count
    mse_tally /= batch_count
    elbo_tally /= batch_count
    nll_tally /= batch_count
    for key in kl_all_tallies:
        kl_all_tallies[key] /= batch_count
    if hyper_params["likelihood"] == "Gaussian" and hyper_params["predict_x_var"]:
        std_mean_tally /= batch_count
        std_std_tally /= batch_count

    if validation_mask is not None:
        # bits per dim: VLB/(hwc log(2))
        elbo_tally_in_bits_per_dim = -elbo_tally / (
            torch.count_nonzero(validation_mask).item() * np.log(2.0)
        )
        if hyper_params["data_is_colour"]:
            elbo_tally_in_bits_per_dim = elbo_tally_in_bits_per_dim / 3

        nll_tally_in_bits_per_dim = -nll_tally / (
            torch.count_nonzero(validation_mask).item() * np.log(2.0)
        )
        if hyper_params["data_is_colour"]:
            nll_tally_in_bits_per_dim = nll_tally_in_bits_per_dim / 3
    else:
        # bits per dim: VLB/(hwc log(2))
        elbo_tally_in_bits_per_dim = -elbo_tally / (
            np.prod(hyper_params["data_shape"]) * np.log(2.0)
        )
        if hyper_params["data_is_colour"]:
            elbo_tally_in_bits_per_dim = elbo_tally_in_bits_per_dim / 3

        nll_tally_in_bits_per_dim = -nll_tally / (
            np.prod(hyper_params["data_shape"]) * np.log(2.0)
        )
        if hyper_params["data_is_colour"]:
            nll_tally_in_bits_per_dim = nll_tally_in_bits_per_dim / 3

    loss_string = summary_text_prefix + " ELBO: {:.4f}, MSE: {:.4f}, KL: {:.4f}".format(
        elbo_tally, mse_tally, kl_tally
    )
    misc.print_0(hyper_params, loss_string)

    if hyper_params["local_rank"] == 0:
        for key in gradient_norms:
            current_list = gradient_norms[key]
            if len(current_list) > 0:
                data = [
                    a for _, a in current_list if np.sum(np.isnan(a) + np.isinf(a)) == 0
                ]
                if len(data) == 0:
                    m = 0
                else:
                    m = int(np.mean(data))
                text = "Average (non-NaN) grad norm (" + key + "): {:.1f}".format(m)
                misc.print_0(hyper_params, text)

    if hyper_params["local_rank"] == 0:
        """
        Should prob only use the TB writer on global_rank 0...
        """
        writer.add_scalar(writer_prefix + "_loss", loss_tally, epoch)
        writer.add_scalar(writer_prefix + "_kl", kl_tally, epoch)
        writer.add_scalar(writer_prefix + "_mse", mse_tally, epoch)
        writer.add_scalar(
            writer_prefix + "_elbo_tally_in_bits_per_dim",
            elbo_tally_in_bits_per_dim,
            epoch,
        )
        writer.add_scalar(
            writer_prefix + "_nll_tally_in_bits_per_dim",
            nll_tally_in_bits_per_dim,
            epoch,
        )

    output_dict = {
        "loss_tally": loss_tally,
        "validation_mask": validation_mask,
        "kl_tally": kl_tally,
        "mse_tally": mse_tally,
        "kl_all_tallies": kl_all_tallies,
        "nll_tally_in_bits_per_dim": nll_tally_in_bits_per_dim,
        "gradient_norms": gradient_norms,
    }

    return output_dict
