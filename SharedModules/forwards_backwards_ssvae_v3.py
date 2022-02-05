import misc
import torch
import numpy as np


class up_down():
    """
    This is the bottom-up then top-down, complete forward pass for the very deep VAE, plus the computation of the loss
    at the end, and gradients
    :param kwargs:
    :return:
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.bottom_up_graph_1 = kwargs["bottom_up_graph_1"]
        self.bottom_up_graph_2 = kwargs["bottom_up_graph_2"]
        self.top_down_graph = kwargs["top_down_graph"]
        self.hyper_params = kwargs["hyper_params"]
        self.training = kwargs["training"]
        self.params = kwargs["params"]  # Always needed because the loss includes l2 regularisation
        self.device = kwargs["device"]
        self.ddp_no_sync = kwargs['ddp_no_sync']
        self.num_imaging_modalities = kwargs['num_imaging_modalities']

        if self.training:
            self.retain_graph = kwargs["retain_graph"]
            self.scaler = kwargs["scaler"]

    def forward(self, **kwargs):
        p_input = kwargs['p_input']
        q_input = kwargs['q_input']
        current_flags = kwargs['current_flags']
        current_non_imaging = kwargs['current_non_imaging']  # If None, must concat to the flag

        if 'output_distribution' in kwargs:
            output_distribution = kwargs['output_distribution']
        else:
            output_distribution = 'Gaussian'

        if 'loss_multiplier' in kwargs:
            loss_multiplier = kwargs['loss_multiplier']
        else:
            loss_multiplier = 1

        """
        Posterior bottom-up
        Add option to concat posterior_input_channels at multiple depths
        """
        input_dictionary_1 = {'data': q_input, 'flag': current_flags}
        with self.bottom_up_graph_1.model.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
            data_dictionary_1 = self.bottom_up_graph_1.model(input_dictionary_1)
        data_dictionary = {'data': data_dictionary_1['data'], 'KL_list': []}
        for key in data_dictionary_1:
            data_dictionary['encoder1_' + key] = data_dictionary_1[key]

        """
        Prior bottom-up
        """
        if p_input is None:
            # This tells the following model that we are not conditioning the prior on anything.
            data_dictionary['encoder2_data'] = None
        else:
            input_dictionary_2 = {'data': p_input}
            with self.bottom_up_graph_2.model.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
                data_dictionary_2 = self.bottom_up_graph_2.model(input_dictionary_2)
            for key in data_dictionary_2:
                # Copy across the prior embeddings to the main dictionary
                data_dictionary['encoder2_' + key] = data_dictionary_2[key]

        """
        Make sure I can concatenate the flag with the embedding
        When sampling unconditonally I still need to be able to specify which distribution to sample from, so the
        flag here is essential.
        """

        data_dictionary['flag'] = current_flags
        current_flags = misc.expand_flag_v2(current_flags, self.device, self.hyper_params['use_nii_data'],
                                            q_input.shape[0])
        data_dictionary['expanded_flag'] = current_flags

        if current_non_imaging is not None:
            for _ in range(3):
                current_non_imaging = torch.unsqueeze(current_non_imaging, -1)
            if self.hyper_params['use_nii_data']:
                current_non_imaging = torch.unsqueeze(current_non_imaging, -1)
            data_dictionary['non_imaging_data'] = current_non_imaging

        if 'res_to_sample_from_prior' in kwargs:
            # This is for producing reconstructions where certain latents are sampled from the prior
            data_dictionary['res_to_sample_from_prior'] = kwargs['res_to_sample_from_prior']

        with self.top_down_graph.latents.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
            data_dictionary_latents = self.top_down_graph.latents(data_dictionary)
        with self.top_down_graph.x_mu.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
            data_dictionary_x_mu = self.top_down_graph.x_mu(data_dictionary_latents)

        if self.hyper_params['separate_output_loc_scale_convs']:
            num_modalities = int(data_dictionary_x_mu['data'].shape[1])
        else:
            num_modalities = int(0.5 * data_dictionary_x_mu['data'].shape[1])

        if output_distribution == 'Gaussian':
            x_mu, x_std, x_var, x_log_var = misc.gaussian_output(data_dictionary_x_mu, self.top_down_graph,
                                                                 self.hyper_params, num_modalities=num_modalities)
        elif output_distribution == 'Bernoulli':
            if self.hyper_params['predict_x_var']:
                logits = data_dictionary_x_mu['data'][:, 0:num_modalities, ...]
            else:
                logits = data_dictionary_x_mu['data']

            probs = logits.detach().clone()
            probs = torch.sigmoid(probs)

            preds = probs.detach().clone()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

        if misc.key_is_true(kwargs, 'reconstruct_only'):
            if output_distribution == 'Gaussian':
                if self.hyper_params['predict_x_var']:
                    if misc.key_is_true(kwargs, 'output_as_numpy'):
                        output = [x.type(torch.float32).cpu().detach().numpy() for x in [x_mu, x_std]]
                    else:
                        output = x_mu, x_std
                else:
                    if misc.key_is_true(kwargs, 'output_as_numpy'):
                        output = x_mu.type(torch.float32).cpu().detach().numpy()
                    else:
                        output = x_mu

                return output

            elif output_distribution == 'Bernoulli':
                if misc.key_is_true(kwargs, 'output_as_numpy'):
                    output = preds.type(torch.float32).cpu().detach().numpy()
                else:
                    output = preds

                return output

        else:
            current_target = kwargs['current_target']
            if output_distribution == 'Gaussian':
                log_likelihood_per_dim, squared_difference = misc.gaussian_likelihood(current_target, x_mu, x_var,
                                                                                      x_log_var, self.hyper_params)
            elif output_distribution == 'Bernoulli':
                log_likelihood_per_dim = \
                    -torch.nn.functional.binary_cross_entropy_with_logits(logits, current_target, reduction='none')

                if 'experimental_XENT_coeff' in self.hyper_params:
                    log_likelihood_per_dim *= self.hyper_params['experimental_XENT_coeff']

                squared_difference = torch.square(current_target - probs)
                sorenson_dice = misc.dice(preds[:, -1, ...], current_target[:, -1, ...])

                # if torch.sum(torch.isnan(sorenson_dice)) > 0:
                #     dims = misc.non_batch_dims(preds[:, -1, ...])
                #
                #     a = current_target[:, -1, ...]
                #     aa = torch.sum(a, dim=[1, 2, 3])
                #
                #     numerator = 2 * torch.sum(preds[:, -1, ...] * current_target[:, -1, ...], dim=dims)
                #     denominator = torch.sum(preds[:, -1, ...], dim=dims) + torch.sum(current_target[:, -1, ...], dim=dims)
                #     sorenson_dice = torch.mean(numerator / denominator)
                #
                #     print("")

            """
            Redact anything that is missing in the input. This completes the reconstruction process.
            """
            left_of_pipe = current_flags[:, 0:num_modalities, ...]
            log_likelihood_per_dim = torch.mul(log_likelihood_per_dim, left_of_pipe)
            squared_difference = torch.mul(squared_difference, left_of_pipe)

            log_likelihood = torch.sum(log_likelihood_per_dim, dim=misc.non_batch_dims(log_likelihood_per_dim))

            kl_all = data_dictionary_latents['KL_list']
            if 'KLs_to_use' in self.hyper_params and self.hyper_params['KLs_to_use']:
                for i, j in enumerate(self.hyper_params['KLs_to_use']):
                    if not j:
                        kl_all[-1 - i] *= 1e-10
            kl = torch.stack(kl_all)

            kl_for_loss = torch.sum(torch.stack(kl_all), 0)
            if 'kl_multiplier' in self.hyper_params and self.hyper_params['kl_multiplier'] == 0:
                kl_for_loss *= 0
                kl *= 0

            kl = torch.sum(kl, 0)
            kl_all = [k.detach() for k in [torch.mean(a) for a in kl_all]]

            vlb = log_likelihood - kl_for_loss

            if 'loss_mask' in kwargs:
                loss = torch.mean(-torch.mul(vlb, kwargs['loss_mask']))
            else:
                loss = torch.mean(-vlb)
            loss += misc.sum_non_bias_l2_norms(self.params, self.hyper_params['l2_reg_coeff'])
            loss *= loss_multiplier

            kl = torch.mean(kl)
            mse = torch.mean(squared_difference)
            elbo = torch.mean(-vlb)

            if self.training:
                # We check the loss for NaNs, and if it's OK we compute the gradient
                nan_count = torch.isnan(loss).sum() + torch.isinf(loss).sum()
                nan_count = nan_count.item()

                if nan_count == 0:
                    self.scaler.scale(loss).backward(retain_graph=self.retain_graph)
            else:
                nan_count = 0

            per_iteration_individual_totals = kwargs['tallies'][0]
            modality = kwargs['tallies'][1]
            modality = misc.sort_dist(modality, self.num_imaging_modalities)


            per_iteration_individual_totals['loss'][modality].append(loss.detach())
            per_iteration_individual_totals['kl'][modality].append(kl.detach())
            per_iteration_individual_totals['mse'][modality].append(mse.detach())
            per_iteration_individual_totals['elbo'][modality].append(elbo.detach())
            per_iteration_individual_totals['kl_all'][modality].append(kl_all)
            per_iteration_individual_totals['nan_count'][modality].append(nan_count)

            if output_distribution == 'Bernoulli':
                per_iteration_individual_totals['dice'][modality].append(sorenson_dice)

            return per_iteration_individual_totals


class sample():
    """
    is_sampling = 'encoder1_lateral_skip_0' not in data_dictionary or \
                          data_dictionary['encoder1_lateral_skip_0'] is None
    :param kwargs:
    :return:
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.bottom_up_graph_2 = kwargs["bottom_up_graph_2"]
        self.top_down_graph = kwargs["top_down_graph"]
        self.hyper_params = kwargs["hyper_params"]
        self.do_impute = kwargs["do_impute"]
        self.training = kwargs['training']
        self.params = kwargs['params']  # Always needed because the imputation loss includes l2 regularisation
        self.device = kwargs['device']
        self.ddp_no_sync = kwargs['ddp_no_sync']

        if self.training:
            self.retain_graph = kwargs["retain_graph"]
            self.scaler = kwargs['scaler']
            self.make_imputation_differentiable = kwargs["make_imputation_differentiable"]

    def forward(self, **kwargs):
        p_input = kwargs['p_input']  # This is *IMAGE* input for the prior...
        current_non_imaging = kwargs['current_non_imaging']  # If None, must concat to the flag
        current_flags = kwargs['current_flags']

        if 'output_distribution' in kwargs:
            output_distribution = kwargs['output_distribution']
        else:
            output_distribution = 'Gaussian'

        if p_input is None:
            batch_size = kwargs['num_samples']
            device = kwargs['device']
        else:
            batch_size = p_input.shape[0]
            device = p_input.device

        if self.hyper_params['use_nii_data']:
            data_dictionary = {'data': torch.zeros((batch_size, self.hyper_params['channels'][-1], 1, 1, 1), device=device)}
        else:
            data_dictionary = {'data': torch.zeros((batch_size, self.hyper_params['channels'][-1], 1, 1), device=device)}

        if p_input is None:
            data_dictionary['encoder2_data'] = None
        else:
            # Prior bottom-up
            input_dictionary_2 = {'data': p_input.clone()}
            with self.bottom_up_graph_2.model.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
                data_dictionary_2 = self.bottom_up_graph_2.model(input_dictionary_2)
            for key in data_dictionary_2:
                data_dictionary['encoder2_' + key] = data_dictionary_2[key]

        data_dictionary['flag'] = current_flags
        current_flags = misc.expand_flag_v2(current_flags, self.device, self.hyper_params['use_nii_data'], batch_size)
        data_dictionary['expanded_flag'] = current_flags

        if current_non_imaging is not None:
            for _ in range(3):
                current_non_imaging = torch.unsqueeze(current_non_imaging, -1)
            if self.hyper_params['use_nii_data']:
                current_non_imaging = torch.unsqueeze(current_non_imaging, -1)
            data_dictionary['non_imaging_data'] = current_non_imaging

        with self.top_down_graph.latents.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
            data_dictionary_latents = self.top_down_graph.latents(data_dictionary)
        with self.top_down_graph.x_mu.no_sync() if self.ddp_no_sync else misc.dummy_context_mgr():
            data_dictionary_x_mu = self.top_down_graph.x_mu(data_dictionary_latents)

        if self.hyper_params['separate_output_loc_scale_convs']:
            num_modalities = int(data_dictionary_x_mu['data'].shape[1])
        else:
            num_modalities = int(0.5 * data_dictionary_x_mu['data'].shape[1])

        if output_distribution == 'Gaussian':
            x_mu, x_std, x_var, x_log_var = misc.gaussian_output(data_dictionary_x_mu, self.top_down_graph,
                                                                 self.hyper_params, num_modalities=num_modalities)
        elif output_distribution == 'Bernoulli':
            if self.hyper_params['predict_x_var']:
                logits = data_dictionary_x_mu['data'][:, 0:num_modalities, ...]
            else:
                logits = data_dictionary_x_mu['data']

            probs = logits.clone()
            probs = torch.sigmoid(probs)
            predictions = probs.clone()
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0

        if self.do_impute:
            """
            This is the code for imputing during a forward pass. It gives us the entropy loss too.
            """

            if 'loss_multiplier' in kwargs:
                loss_multiplier = kwargs['loss_multiplier']
            else:
                loss_multiplier = 1

            if output_distribution == 'Gaussian':
                x_mu_to_use = x_mu
            else:
                x_mu_to_use = predictions

            if not (self.training and self.make_imputation_differentiable):
                x_mu_to_use = x_mu_to_use.detach()

            left_of_pipe = current_flags[:, 0:num_modalities, ...]
            current_batch_with_imputation = torch.mul(x_mu_to_use, left_of_pipe) + torch.mul(p_input, 1 - left_of_pipe)

            nan_count = 0

            if misc.key_is_true(self.hyper_params, 'veto_imputation_loss'):
                if output_distribution == 'Gaussian':
                    loss = torch.Tensor([0]).to(x_mu.device)
                else:
                    loss = torch.Tensor([0]).to(logits.device)
            else:
                if output_distribution == 'Gaussian':
                    if self.hyper_params['predict_x_var']:
                        loss = 0.5 * torch.mul(np.log(2 * np.pi) + x_log_var + 1, left_of_pipe)
                        loss = torch.sum(loss, dim=misc.non_batch_dims(loss))
                        if 'loss_mask' in kwargs:
                            loss = torch.mul(loss, kwargs['loss_mask'])
                        loss = torch.mean(loss)
                        loss += misc.sum_non_bias_l2_norms(self.params, self.hyper_params['l2_reg_coeff'])
                        loss *= loss_multiplier

                        if self.training:
                            # We check the loss for NaNs, and if it's OK we compute the gradient
                            nan_count = torch.isnan(loss).sum() + torch.isinf(loss).sum()
                            nan_count = nan_count.item()
                            if nan_count == 0:
                                self.scaler.scale(loss).backward(retain_graph=self.retain_graph)
                    else:
                        loss = 0.5 * torch.mul(np.log(2 * np.pi) + torch.zeros_like(x_mu) + 1, left_of_pipe)
                        loss = torch.sum(loss, dim=misc.non_batch_dims(loss))
                        if 'loss_mask' in kwargs:
                            loss = torch.mul(loss, kwargs['loss_mask'])
                        loss = torch.mean(loss)
                        loss += misc.sum_non_bias_l2_norms(self.params, self.hyper_params['l2_reg_coeff'])
                        loss *= loss_multiplier

                elif output_distribution == 'Bernoulli':
                    # -p x log(p) - (1 - p) x log(1 - p)
                    loss = - probs * logits - (1 - probs) * torch.log(1 - probs + 1e-5)
                    loss = torch.sum(loss, dim=misc.non_batch_dims(loss))
                    if 'loss_mask' in kwargs:
                        loss = torch.mul(loss, kwargs['loss_mask'])
                    loss = torch.mean(loss)
                    loss += misc.sum_non_bias_l2_norms(self.params, self.hyper_params['l2_reg_coeff'])
                    loss *= loss_multiplier

                    if self.training:
                        # We check the loss for NaNs, and if it's OK we compute the gradient
                        nan_count = torch.isnan(loss).sum() + torch.isinf(loss).sum()
                        nan_count = nan_count.item()
                        if nan_count == 0:
                            self.scaler.scale(loss).backward(retain_graph=self.retain_graph)

            output = (current_batch_with_imputation,)

            if 'tallies' in kwargs:
                per_iteration_individual_totals = kwargs['tallies'][0]
                modality = kwargs['tallies'][1]
                per_iteration_individual_totals['loss'][modality].append(loss.detach())
                per_iteration_individual_totals['entropy'][modality].append(loss.detach())
                per_iteration_individual_totals['nan_count'][modality].append(nan_count)
                output += (per_iteration_individual_totals,)

        else:
            if output_distribution == 'Gaussian':
                if self.hyper_params['predict_x_var']:
                    if misc.key_is_true(kwargs, 'output_as_numpy'):
                        output = [x.type(torch.float32).cpu().detach().numpy() for x in [x_mu, x_std, x_var, x_log_var]]
                    else:
                        output = x_mu, x_std, x_var, x_log_var
                else:
                    if misc.key_is_true(kwargs, 'output_as_numpy'):
                        output = x_mu.type(torch.float32).cpu().detach().numpy()
                    else:
                        output = x_mu

            elif output_distribution == 'Bernoulli':
                if misc.key_is_true(kwargs, 'output_as_numpy'):
                    output = predictions.type(torch.float32).cpu().detach().numpy()
                else:
                    output = predictions

        return output
