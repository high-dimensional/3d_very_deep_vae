import torch
import torch.nn as nn
import torch.cuda.amp as amp
from ..misc import misc
import numpy as np


class TopDownBlock(nn.Module):
    """
    Top down block for very deep VAE (without the residual block, which I've given its own class)
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.kwargs = kwargs
        self.channels_in = self.kwargs["channels_in"]
        self.channels_out = self.kwargs["channels_out"]

        if "channels_hidden" in self.kwargs:
            self.channels_hidden = self.kwargs["channels_hidden"]
        else:
            self.channels_hidden = self.kwargs["channels_out"]

        self.lateral_connection_channels = self.kwargs["lateral_connection_channels"]
        self.lateral_skip_con_to_use = self.kwargs["lateral_skip_con_to_use"]
        self.variance_bounds = self.kwargs["variance_bounds"][:]
        self.precision_reweighting = self.kwargs["precision_reweighting"]
        self.separate_loc_scale_convs = self.kwargs["separate_loc_scale_convs"]
        self.hyper_params = self.kwargs["hyper_params"]
        self.hidden_kernel_size = self.kwargs["hidden_kernel_size"]
        self.param_count = 0
        out_txt = ""

        # New counter to power the mechnism for reducing the upper clamp limit on the std linearly down to a minimum
        # upper limit
        if "minimum_upper_hidden_bound" in self.hyper_params:
            self.index_of_latent = self.kwargs["index_of_latent"]
            self.biggest_latent_index = (
                np.sum(self.hyper_params["latent_feature_maps_per_resolution"]) - 1
            )
            self.minimum_upper_variance_bound = self.hyper_params[
                "minimum_upper_hidden_bound"
            ]
            if "data_is_3d" in self.hyper_params and self.hyper_params["data_is_3d"]:
                self.variance_bounds[1] = (
                    self.variance_bounds[1]
                    * (
                        (self.biggest_latent_index - self.index_of_latent)
                        / self.biggest_latent_index
                    )
                    ** 3
                    + self.minimum_upper_variance_bound
                    * (self.index_of_latent / self.biggest_latent_index) ** 3
                )
            else:
                self.variance_bounds[1] = (
                    self.variance_bounds[1]
                    * (
                        (self.biggest_latent_index - self.index_of_latent)
                        / self.biggest_latent_index
                    )
                    ** 2
                    + self.minimum_upper_variance_bound
                    * (self.index_of_latent / self.biggest_latent_index) ** 2
                )

        if "length_of_flag" in self.kwargs:
            self.length_of_flag = self.kwargs["length_of_flag"]
        else:
            self.length_of_flag = 0

        if "non_imaging_dims" in self.kwargs:
            self.non_imaging_dims = self.kwargs["non_imaging_dims"]
        else:
            self.non_imaging_dims = 0

        if "normalise_weight_by_depth" in self.kwargs:
            self.normalise_weight_by_depth = self.kwargs["normalise_weight_by_depth"]
        else:
            self.normalise_weight_by_depth = False

        if "depth_override" in self.hyper_params:
            self.depth = self.hyper_params["depth_override"]
        else:
            self.depth = np.sum(self.hyper_params["latent_feature_maps_per_resolution"])
            self.depth += 2 * (len(self.hyper_params["channels"]) - 1)

        if misc.key_is_true(self.kwargs, "conditional_prior"):
            self.conditional_model = True
            # print("")
        else:
            if misc.key_is_true(self.hyper_params, "conditional_model"):
                self.conditional_model = True
            else:
                self.conditional_model = False

        if (
            "channels_for_latent" in self.kwargs
            and self.kwargs["channels_for_latent"] < self.channels_out
        ):
            self.channels_for_latent = self.kwargs["channels_for_latent"]
        else:
            self.channels_for_latent = self.channels_out

        if "shared_group_ops" in self.kwargs:
            self.shared_group_ops = self.kwargs["shared_group_ops"]
        else:
            self.shared_group_ops = None

        if misc.key_is_true(self.hyper_params, "veto_noise_injection"):
            self.inject_noise = False
        else:
            self.inject_noise = True

        if "hidden_kernel_size" in self.kwargs:
            self.hidden_kernel_size = self.kwargs["hidden_kernel_size"]
            if self.hidden_kernel_size == 1:
                self.pad = nn.Identity()
            elif self.hidden_kernel_size == 2:
                if (
                    "data_is_3d" in self.hyper_params
                    and self.hyper_params["data_is_3d"]
                ):
                    self.pad = nn.ConstantPad2d((1, 0, 1, 0, 1, 0), 0)
                else:
                    self.pad = nn.ConstantPad2d((1, 0, 1, 0), 0)
            else:
                if (
                    "data_is_3d" in self.hyper_params
                    and self.hyper_params["data_is_3d"]
                ):
                    self.pad = nn.ConstantPad2d((1, 1, 1, 1, 1, 1), 0)
                else:
                    self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)
        else:
            if "data_is_3d" in self.hyper_params and self.hyper_params["data_is_3d"]:
                self.pad = nn.ConstantPad2d((1, 1, 1, 1, 1, 1), 0)
            else:
                self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)

        if "data_is_3d" in self.hyper_params and self.hyper_params["data_is_3d"]:
            ConvOp = nn.Conv3d
            BatchNormOp = nn.BatchNorm3d
            d = 3
            filter_space_dims = (
                str(self.hidden_kernel_size)
                + "x"
                + str(self.hidden_kernel_size)
                + "x"
                + str(self.hidden_kernel_size)
            )
            one_dims = "1x1x1"
        else:
            ConvOp = nn.Conv2d
            BatchNormOp = nn.BatchNorm2d
            d = 2
            filter_space_dims = (
                str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size)
            )
            one_dims = "1x1"

        if "activation" in self.kwargs:
            self.activation = self.kwargs["activation"]
        else:
            self.activation = torch.nn.GELU()

        # self.use_batch_norm = False
        if ("veto_batch_norm" in self.kwargs and self.kwargs["veto_batch_norm"]) or (
            "veto_batch_norm" in self.hyper_params
            and self.hyper_params["veto_batch_norm"]
        ):
            self.use_batch_norm = False
        else:
            self.use_batch_norm = True
            self.batch_norm = BatchNormOp(
                num_features=self.channels_out,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
            self.param_count += 2 * self.channels_out

        if "uppermost_block" in self.kwargs and self.kwargs["uppermost_block"]:
            self.uppermost_block = True
            self.use_skip_con = False

            # self.use_uppermost_skip_con = True
            # if not self.channels_in == self.channels_out:
            #     self.uppermost_skip_con_conv = ConvOp(in_channels=self.channels_in, out_channels=self.channels_out,
            #                                 kernel_size=1, stride=1, padding=0, bias=True)
            #
            #     self.param_count += self.channels_in * self.channels_out
            #     out_txt += "> uppermost_skip: " + one_dims + "x" + str(self.channels_in) + "x" + str(self.channels_out) \
            #                + "\n"
        else:
            self.use_uppermost_skip_con = False
            self.uppermost_block = False

            if "veto_skip_con" in self.kwargs and self.kwargs["veto_skip_con"]:
                self.use_skip_con = False
            else:
                self.use_skip_con = True
                if hasattr(self.shared_group_ops, "skip_con_conv"):
                    self.skip_con_conv = self.shared_group_ops.skip_con_conv
                else:
                    if not self.channels_in == self.channels_out:
                        self.skip_con_conv = ConvOp(
                            in_channels=self.channels_in,
                            out_channels=self.channels_out,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )

                        self.param_count += self.channels_in * self.channels_out
                        out_txt += (
                            "> skip: "
                            + one_dims
                            + "x"
                            + str(self.channels_in)
                            + "x"
                            + str(self.channels_out)
                            + "\n"
                        )

        if self.uppermost_block:
            # Predict posterior from lateral skip connection only
            dims_into_prior_predicter = self.channels_in
        else:
            # Predict posterior from lateral skip connection AND incoming data from previous block
            if self.conditional_model:
                # In this case we are also taking in data from bottom_up encoder 2
                dims_into_prior_predicter = (
                    self.lateral_connection_channels + self.channels_in
                )
            else:
                dims_into_prior_predicter = self.channels_in

        dims_into_prior_predicter += self.length_of_flag
        dims_into_prior_predicter += self.non_imaging_dims

        if self.separate_loc_scale_convs:
            # Use completely separate parameters to predict the location and scale
            if hasattr(self.shared_group_ops, "convs_p_mu") and hasattr(
                self.shared_group_ops, "convs_p_log_var"
            ):
                self.convs_p_mu = self.shared_group_ops.convs_p_mu
                self.convs_p_log_var = self.shared_group_ops.convs_p_log_var
            else:
                if not (
                    self.uppermost_block and not self.conditional_model
                ):  # Prior not predicted from data in uppermost block when not conditional model
                    # self.convs_p_mu = nn.ModuleList()
                    # self.convs_p_log_var = nn.ModuleList()
                    # self.convs_p_mu.append(ConvOp(in_channels=dims_into_prior_predicter, out_channels=self.channels_hidden,
                    #                               kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                    # self.convs_p_mu.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                    #                               kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                    # self.convs_p_log_var.append(ConvOp(in_channels=dims_into_prior_predicter, out_channels=self.channels_hidden,
                    #                                    kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                    #                                    bias=True))
                    # self.convs_p_log_var.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                    #                                    kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                    #                                    bias=True))
                    #
                    # self.param_count += 2 * self.hidden_kernel_size ** d * dims_into_prior_predicter * self.channels_hidden
                    # self.param_count += 2 * self.hidden_kernel_size ** d * self.channels_hidden * self.channels_for_latent
                    # out_txt += "> p(z): " + str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size) + "x" + \
                    #                  str(dims_into_prior_predicter) + "x" + str(self.channels_hidden)
                    # out_txt += ", " + str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size) + "x" + \
                    #                  str(self.channels_hidden) + "x" + str(self.channels_for_latent)
                    # out_txt += ", " + str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size) + "x" + \
                    #                  str(dims_into_prior_predicter) + "x" + str(self.channels_hidden)
                    # out_txt += ", " + str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size) + "x" + \
                    #                  str(self.channels_hidden) + "x" + str(self.channels_for_latent) + "\n"

                    self.convs_p_mu = nn.ModuleList()
                    self.convs_p_log_var = nn.ModuleList()
                    self.convs_p_mu.append(
                        ConvOp(
                            in_channels=dims_into_prior_predicter,
                            out_channels=self.channels_hidden,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p_mu.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_hidden,
                            kernel_size=self.hidden_kernel_size,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p_mu.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_hidden,
                            kernel_size=self.hidden_kernel_size,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p_mu.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_for_latent,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )

                    self.convs_p_log_var.append(
                        ConvOp(
                            in_channels=dims_into_prior_predicter,
                            out_channels=self.channels_hidden,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p_log_var.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_hidden,
                            kernel_size=self.hidden_kernel_size,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p_log_var.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_hidden,
                            kernel_size=self.hidden_kernel_size,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p_log_var.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_for_latent,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )

                    # In the very deep VAE paper
                    self.convs_p_mu[-1].weight.data *= 0.0
                    self.convs_p_log_var[-1].weight.data *= 0.0

                    self.param_count += (
                        2 * dims_into_prior_predicter * self.channels_hidden
                    )
                    self.param_count += (
                        4
                        * self.hidden_kernel_size**d
                        * self.channels_hidden
                        * self.channels_hidden
                    )
                    self.param_count += (
                        2 * self.channels_hidden * self.channels_for_latent
                    )

                    out_txt += (
                        "> p(z): "
                        + one_dims
                        + "x"
                        + str(dims_into_prior_predicter)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + filter_space_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + filter_space_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + one_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_for_latent)
                    )

                    out_txt += (
                        "; "
                        + one_dims
                        + "x"
                        + str(dims_into_prior_predicter)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + filter_space_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + filter_space_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + one_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_for_latent)
                        + "\n"
                    )

        else:
            # Mostly share the parameters to predict the location and scale
            if hasattr(self.shared_group_ops, "convs_p"):
                self.convs_p = self.shared_group_ops.convs_p
            else:
                if not (
                    self.uppermost_block and not self.conditional_model
                ):  # Prior not predicted from data in uppermost block when not conditional model
                    # self.convs_p = nn.ModuleList()
                    # self.convs_p.append(ConvOp(in_channels=dims_into_prior_predicter, out_channels=self.channels_hidden,
                    #                            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                    # self.convs_p.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_hidden,
                    #                            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                    # self.convs_p.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                    #                            kernel_size=1, stride=1, padding=0, bias=True))
                    # self.convs_p.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                    #                            kernel_size=1, stride=1, padding=0, bias=True))
                    #
                    # self.param_count += self.hidden_kernel_size ** d * dims_into_prior_predicter * self.channels_hidden
                    # self.param_count += self.hidden_kernel_size ** d * self.channels_hidden * self.channels_hidden
                    # self.param_count += 2 * self.channels_hidden * self.channels_for_latent
                    # out_txt += "> p(z): " + filter_space_dims + "x" + str(dims_into_prior_predicter) + "x" + str(self.channels_hidden)
                    # out_txt += ", " + filter_space_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_hidden)
                    # out_txt += ", " + one_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_for_latent)
                    # out_txt += ", " + one_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_for_latent) + "\n"

                    self.convs_p = nn.ModuleList()
                    self.convs_p.append(
                        ConvOp(
                            in_channels=dims_into_prior_predicter,
                            out_channels=self.channels_hidden,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_hidden,
                            kernel_size=self.hidden_kernel_size,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_hidden,
                            kernel_size=self.hidden_kernel_size,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_for_latent,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )
                    self.convs_p.append(
                        ConvOp(
                            in_channels=self.channels_hidden,
                            out_channels=self.channels_for_latent,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    )

                    # self.convs_p_bn = nn.ModuleList()
                    # self.convs_p_bn.append(BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
                    #                               track_running_stats=True))
                    # self.convs_p_bn.append(BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
                    #                               track_running_stats=True))
                    # self.convs_p_bn.append(BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
                    #                               track_running_stats=True))

                    self.convs_p[-1].weight.data *= 0.0
                    self.convs_p[-2].weight.data *= 0.0

                    self.param_count += dims_into_prior_predicter * self.channels_hidden
                    self.param_count += (
                        2
                        * self.hidden_kernel_size**d
                        * self.channels_hidden
                        * self.channels_hidden
                    )
                    self.param_count += (
                        2 * self.channels_hidden * self.channels_for_latent
                    )
                    out_txt += (
                        "> p(z): "
                        + one_dims
                        + "x"
                        + str(dims_into_prior_predicter)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + filter_space_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + filter_space_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_hidden)
                    )
                    out_txt += (
                        ", "
                        + one_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_for_latent)
                    )
                    out_txt += (
                        ", "
                        + one_dims
                        + "x"
                        + str(self.channels_hidden)
                        + "x"
                        + str(self.channels_for_latent)
                        + "\n"
                    )

        if self.conditional_model or not self.uppermost_block:
            # Add this to the parameter tally!!!
            self.convs_p_skip_conv_after_prior = ConvOp(
                in_channels=self.channels_hidden,
                out_channels=self.channels_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

            if "zero_biases" in self.hyper_params and self.hyper_params["zero_biases"]:
                self.convs_p_skip_conv_after_prior.weight.data *= (
                    0.0  # In the v deep VAE paper
                )
                self.convs_p_skip_conv_after_prior.bias.data *= 0.0

            out_txt += (
                "> skip_p: "
                + one_dims
                + "x"
                + str(self.channels_hidden)
                + "x"
                + str(self.channels_out)
                + "\n"
            )

        if self.uppermost_block:
            # Predict posterior from lateral skip connection only
            dims_into_posterior_predicter = self.lateral_connection_channels
        else:
            # Predict posterior from lateral skip connection AND incoming data from previous block
            dims_into_posterior_predicter = (
                self.lateral_connection_channels + self.channels_in
            )

        dims_into_posterior_predicter += self.length_of_flag
        dims_into_posterior_predicter += self.non_imaging_dims

        if self.separate_loc_scale_convs:
            if hasattr(self.shared_group_ops, "convs_q_mu") and hasattr(
                self.shared_group_ops, "convs_q_log_var"
            ):
                self.convs_q_mu = self.shared_group_ops.convs_q_mu
                self.convs_q_log_var = self.shared_group_ops.convs_q_log_var
            else:
                self.convs_q_mu = nn.ModuleList()
                self.convs_q_log_var = nn.ModuleList()

                # self.convs_q_mu.append(
                #     ConvOp(in_channels=dims_into_posterior_predicter, out_channels=self.channels_hidden,
                #            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                # self.convs_q_mu.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                #                               kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                # self.convs_q_log_var.append(ConvOp(in_channels=dims_into_posterior_predicter,
                #                                    out_channels=self.channels_hidden,
                #                                    kernel_size=self.hidden_kernel_size,
                #                                    stride=1, padding=0, bias=True))
                # self.convs_q_log_var.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                #                                    kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                #                                    bias=True))
                #
                # self.param_count += 2 * self.hidden_kernel_size ** d * dims_into_posterior_predicter * self.channels_hidden
                # self.param_count += 2 * self.hidden_kernel_size ** d * self.channels_hidden * self.channels_for_latent
                #
                # out_txt += "> q(z): " + str(self.hidden_kernel_size) + "x" + str(
                #     self.hidden_kernel_size) + "x" + \
                #                        str(dims_into_posterior_predicter) + "x" + str(self.channels_hidden)
                # out_txt += ", " + str(self.hidden_kernel_size) + "x" + str(
                #     self.hidden_kernel_size) + "x" + \
                #                        str(self.channels_hidden) + "x" + str(self.channels_for_latent)
                # out_txt += ", " + str(self.hidden_kernel_size) + "x" + str(
                #     self.hidden_kernel_size) + "x" + \
                #                        str(dims_into_posterior_predicter) + "x" + str(self.channels_hidden)
                # out_txt += ", " + str(self.hidden_kernel_size) + "x" + str(
                #     self.hidden_kernel_size) + "x" + \
                #                        str(self.channels_hidden) + "x" + str(self.channels_for_latent) + "\n"

                self.convs_q_mu.append(
                    ConvOp(
                        in_channels=dims_into_posterior_predicter,
                        out_channels=self.channels_hidden,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q_mu.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_hidden,
                        kernel_size=self.hidden_kernel_size,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q_mu.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_hidden,
                        kernel_size=self.hidden_kernel_size,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q_mu.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_for_latent,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )

                self.convs_q_log_var.append(
                    ConvOp(
                        in_channels=dims_into_posterior_predicter,
                        out_channels=self.channels_hidden,
                        kernel_size=self.hidden_kernel_size,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q_log_var.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_hidden,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q_log_var.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_hidden,
                        kernel_size=self.hidden_kernel_size,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q_log_var.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_for_latent,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )

                self.param_count += 2 * dims_into_prior_predicter * self.channels_hidden
                self.param_count += (
                    4
                    * self.hidden_kernel_size**d
                    * self.channels_hidden
                    * self.channels_hidden
                )
                self.param_count += 2 * self.channels_hidden * self.channels_for_latent

                out_txt += (
                    "> q(z): "
                    + one_dims
                    + "x"
                    + str(dims_into_prior_predicter)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + filter_space_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + filter_space_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + one_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_for_latent)
                )

                out_txt += (
                    "; "
                    + one_dims
                    + "x"
                    + str(dims_into_prior_predicter)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + filter_space_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + filter_space_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + one_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_for_latent)
                    + "\n"
                )

        else:
            if hasattr(self.shared_group_ops, "convs_q"):
                self.convs_q = self.shared_group_ops.convs_q
            else:
                # self.convs_q = nn.ModuleList()
                # self.convs_q.append(ConvOp(in_channels=dims_into_posterior_predicter, out_channels=self.channels_hidden,
                #            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                # self.convs_q.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_hidden,
                #                            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
                # self.convs_q.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                #                            kernel_size=1, stride=1, padding=0, bias=True))
                # self.convs_q.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_for_latent,
                #                            kernel_size=1, stride=1, padding=0, bias=True))
                #
                # self.param_count += self.hidden_kernel_size ** d * dims_into_posterior_predicter * self.channels_hidden
                # self.param_count += self.hidden_kernel_size ** d * self.channels_hidden * self.channels_hidden
                # self.param_count += 2 * self.channels_hidden * self.channels_for_latent
                #
                # out_txt += "> q(z): " + filter_space_dims + "x" + str(dims_into_posterior_predicter) + "x" + str(self.channels_hidden)
                # out_txt += ", " + filter_space_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_hidden)
                # out_txt += ", " + one_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_for_latent)
                # out_txt += ", " + one_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_for_latent) + "\n"

                self.convs_q = nn.ModuleList()
                self.convs_q.append(
                    ConvOp(
                        in_channels=dims_into_posterior_predicter,
                        out_channels=self.channels_hidden,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_hidden,
                        kernel_size=self.hidden_kernel_size,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_hidden,
                        kernel_size=self.hidden_kernel_size,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_for_latent,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
                self.convs_q.append(
                    ConvOp(
                        in_channels=self.channels_hidden,
                        out_channels=self.channels_for_latent,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )

                # self.convs_q_bn = nn.ModuleList()
                # self.convs_q_bn.append(BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
                #                                 track_running_stats=True))
                # self.convs_q_bn.append(BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
                #                                 track_running_stats=True))
                # self.convs_q_bn.append(BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
                #                                 track_running_stats=True))

                self.param_count += dims_into_posterior_predicter * self.channels_hidden
                self.param_count += (
                    2
                    * self.hidden_kernel_size**d
                    * self.channels_hidden
                    * self.channels_hidden
                )
                self.param_count += 2 * self.channels_hidden * self.channels_for_latent

                out_txt += (
                    "> q(z): "
                    + one_dims
                    + "x"
                    + str(dims_into_posterior_predicter)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + filter_space_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + filter_space_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_hidden)
                )
                out_txt += (
                    ", "
                    + one_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_for_latent)
                )
                out_txt += (
                    ", "
                    + one_dims
                    + "x"
                    + str(self.channels_hidden)
                    + "x"
                    + str(self.channels_for_latent)
                    + "\n"
                )

        # if not self.channels_for_latent == self.channels_out:
        if hasattr(self.shared_group_ops, "conv_z"):
            self.conv_z = self.shared_group_ops.conv_z
        else:
            self.conv_z = ConvOp(
                in_channels=self.channels_for_latent
                + self.length_of_flag
                + self.non_imaging_dims,
                out_channels=self.channels_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

            if "zero_biases" in self.hyper_params and self.hyper_params["zero_biases"]:
                self.conv_z.bias.data *= 0.0

            if self.normalise_weight_by_depth:
                self.conv_z.weight.data *= np.sqrt(1 / self.depth)

            self.param_count += self.channels_for_latent * self.channels_out
            out_txt += (
                "> conv_z: "
                + one_dims
                + "x"
                + str(self.channels_for_latent)
                + "x"
                + str(self.channels_out)
            )

        if "verbose" in self.hyper_params and self.hyper_params["verbose"]:
            print(
                "Top-down block kernels ("
                + str(self.param_count)
                + "): "
                + "\n"
                + out_txt
            )

        if "zero_biases" in self.hyper_params and self.hyper_params["zero_biases"]:
            # if not self.channels_for_latent == self.channels_out:
            if hasattr(self, "conv_z"):
                self.conv_z.bias.data *= 0.0

            if hasattr(self, "convs_p_mu"):
                for a in self.convs_p_mu:
                    a.bias.data *= 0.0

                for a in self.convs_p_log_var:
                    a.bias.data *= 0.0

            if hasattr(self, "convs_p"):
                for a in self.convs_p:
                    a.bias.data *= 0.0

            if hasattr(self, "convs_q_mu"):
                for a in self.convs_q_mu:
                    a.bias.data *= 0.0

                for a in self.convs_q_log_var:
                    a.bias.data *= 0.0

            if hasattr(self, "convs_q"):
                for a in self.convs_q:
                    a.bias.data *= 0.0

        if "use_rezero" in self.hyper_params and self.hyper_params["use_rezero"]:
            self.rezero_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):

            copy_of_incoming_data = data_dictionary[
                "data"
            ].clone()  # For skip connections

            is_sampling = (
                "encoder1_lateral_skip_0" not in data_dictionary
                or data_dictionary["encoder1_lateral_skip_0"] is None
            )

            if is_sampling:
                # SAMPLING MODE
                # Predict prior
                if self.uppermost_block:
                    # When in uppermost block, this must come from the lowest dim b0,b1000
                    if self.conditional_model:
                        if data_dictionary["encoder2_data"] is None:
                            p_data = torch.zeros_like(data_dictionary["data"])
                        else:
                            p_data = data_dictionary["encoder2_data"]
                    copy_of_incoming_data = torch.zeros_like(
                        data_dictionary["data"]
                    )  # Just to be sure
                else:
                    if self.conditional_model:
                        if data_dictionary["encoder2_data"] is None:
                            batch_size = data_dictionary["data"].shape[0]
                            spatial_dims = list(data_dictionary["data"].shape[2:])
                            shape = [
                                batch_size,
                                self.hyper_params["channels"][
                                    self.lateral_skip_con_to_use
                                ],
                            ] + spatial_dims
                            device = data_dictionary["data"].device
                            from_bottom_up_2 = torch.zeros(
                                shape, device=device, dtype=torch.float32
                            )
                        else:
                            from_bottom_up_2 = data_dictionary[
                                "encoder2_lateral_skip_"
                                + str(self.lateral_skip_con_to_use)
                            ]
                        p_data = torch.cat(
                            (from_bottom_up_2, data_dictionary["data"]), dim=1
                        )
                    else:
                        p_data = data_dictionary["data"]

                # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                if self.length_of_flag > 0:
                    """
                    Add the flags to p_data, before it goes through convs
                    """
                    flags = data_dictionary["expanded_flag"]
                    if not self.uppermost_block:
                        if not p_data.shape[2] == flags.shape[2]:
                            if len(flags.shape) == 4:
                                flags = flags.repeat(1, 1, 2, 2)
                            else:
                                flags = flags.repeat(1, 1, 2, 2, 2)
                            data_dictionary["expanded_flag"] = flags
                    p_data = torch.cat((p_data, flags), 1)

                if self.non_imaging_dims > 0:
                    """
                    Add the non_imaging_data to p_data, before it goes through convs
                    """
                    if "non_imaging_data" in data_dictionary:
                        non_imaging_data = data_dictionary["non_imaging_data"]
                    else:
                        batch_size = p_data.shape[0]
                        device = p_data.device
                        if len(p_data.shape) == 4:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1), device=device
                            )
                        else:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1, 1),
                                device=device,
                            )
                    p_data = torch.cat((p_data, non_imaging_data), 1)

                if self.uppermost_block and not self.conditional_model:
                    # Get the shape info from the incoming data

                    # Sample z from N(0,I)
                    latent_shape = list(data_dictionary["data"].shape)
                    latent_shape[1] = self.channels_for_latent
                    data = torch.randn(
                        size=latent_shape, device=data_dictionary["data"].device
                    )

                    if "sampling_noise_std_override" in data_dictionary:
                        data *= data_dictionary["sampling_noise_std_override"]

                    # if self.length_of_flag > 0 and 'flag' in data_dictionary:  # Do I want the second condition?
                    if self.length_of_flag > 0:  # Do I want the second condition?
                        flags = data_dictionary["expanded_flag"]
                        data = torch.cat((data, flags), 1)

                    if self.non_imaging_dims > 0:
                        """
                        Add the non_imaging_data to p_data, before it goes through convs
                        """
                        if "non_imaging_data" in data_dictionary:
                            non_imaging_data = data_dictionary["non_imaging_data"]
                        else:
                            batch_size = data.shape[0]
                            device = data.device
                            if len(data.shape) == 4:
                                non_imaging_data = torch.zeros(
                                    (batch_size, self.non_imaging_dims, 1, 1),
                                    device=device,
                                )
                            else:
                                non_imaging_data = torch.zeros(
                                    (batch_size, self.non_imaging_dims, 1, 1, 1),
                                    device=device,
                                )
                        data = torch.cat((data, non_imaging_data), 1)

                else:
                    # In this case the prior has to be predicted from data
                    if self.separate_loc_scale_convs:
                        p_mu = self.activation(p_data)
                        p_mu = self.convs_p_mu[0](p_mu)
                        p_mu = self.activation(p_mu)
                        p_mu = self.convs_p_mu[1](self.pad(p_mu))
                        p_mu = self.activation(p_mu)
                        p_mu = self.convs_p_mu[2](self.pad(p_mu))
                        p_mu = self.activation(p_mu)
                        p_skip = self.convs_p_skip_conv_after_prior(p_mu)
                        p_mu = self.convs_p_mu[3](p_mu)

                        p_log_var = self.activation(p_data)
                        p_log_var = self.convs_p_log_var[0](p_log_var)
                        p_log_var = self.activation(p_log_var)
                        p_log_var = self.convs_p_log_var[1](self.pad(p_log_var))
                        p_log_var = self.activation(p_log_var)
                        p_log_var = self.convs_p_log_var[2](self.pad(p_log_var))
                        p_log_var = self.activation(p_log_var)
                        p_log_var = self.convs_p_log_var[3](p_log_var)
                    else:
                        p_intermediate = self.activation(p_data)
                        p_intermediate = self.convs_p[0](p_intermediate)
                        p_intermediate = self.activation(p_intermediate)
                        p_intermediate = self.convs_p[1](self.pad(p_intermediate))
                        p_intermediate = self.activation(p_intermediate)
                        p_intermediate = self.convs_p[2](self.pad(p_intermediate))
                        p_intermediate = self.activation(p_intermediate)
                        p_mu = self.convs_p[3](p_intermediate)
                        p_log_var = self.convs_p[4](p_intermediate)
                        p_skip = self.convs_p_skip_conv_after_prior(p_intermediate)

                    if misc.key_is_true(
                        self.hyper_params, "predict_x_scale_with_sigmoid"
                    ):
                        lower = self.variance_bounds[0]
                        upper = self.variance_bounds[1]
                        p_std = lower + (upper - lower) * torch.sigmoid(p_log_var)

                        # Sample z
                        data = p_mu + torch.mul(p_std, torch.randn_like(p_std))
                    else:
                        if self.variance_bounds is not None:
                            p_log_var = torch.clamp(
                                p_log_var,
                                self.variance_bounds[0],
                                self.variance_bounds[1],
                            )

                        # Sample z
                        data = p_mu + torch.mul(
                            torch.exp(0.5 * p_log_var), torch.randn_like(p_log_var)
                        )

                    if "sampling_noise_std_override" in data_dictionary:
                        data *= data_dictionary["sampling_noise_std_override"]

                # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                if self.length_of_flag > 0:
                    flags = data_dictionary["expanded_flag"]
                    if not self.uppermost_block:
                        if not data.shape[2] == flags.shape[2]:
                            # Spatial dims in data have increased, assuming by x2
                            if len(flags.shape) == 4:
                                flags = flags.repeat(1, 1, 2, 2)
                            else:
                                flags = flags.repeat(1, 1, 2, 2, 2)
                            data_dictionary["expanded_flag"] = flags
                    data = torch.cat((data, flags), 1)

                if self.non_imaging_dims > 0:
                    """
                    Add the non_imaging_data to p_data, before it goes through convs
                    """
                    if "non_imaging_data" in data_dictionary:
                        non_imaging_data = data_dictionary["non_imaging_data"]
                    else:
                        batch_size = data.shape[0]
                        device = data.device
                        if len(data.shape) == 4:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1), device=device
                            )
                        else:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1, 1),
                                device=device,
                            )
                    data = torch.cat((data, non_imaging_data), 1)

            else:
                """
                Training mode
                """
                from_bottom_up_1 = data_dictionary[
                    "encoder1_lateral_skip_" + str(self.lateral_skip_con_to_use)
                ]
                if self.conditional_model:
                    if data_dictionary["encoder2_data"] is None:
                        """
                        This is how I use a conditional model, but retain the option of not conditioning on any
                        image data
                        """
                        batch_size = data_dictionary["data"].shape[0]
                        spatial_dims = list(data_dictionary["data"].shape[2:])
                        shape = [
                            batch_size,
                            self.hyper_params["channels"][self.lateral_skip_con_to_use],
                        ] + spatial_dims
                        device = data_dictionary["data"].device
                        from_bottom_up_2 = torch.zeros(
                            shape, device=device, dtype=torch.float32
                        )
                    else:
                        from_bottom_up_2 = data_dictionary[
                            "encoder2_lateral_skip_" + str(self.lateral_skip_con_to_use)
                        ]

                if self.uppermost_block:
                    if self.conditional_model:
                        """
                        Special because the prior gets predicted only from the encoder_2 stream, and because the
                        posterior is predicted only from the incoming encoder_1 stream
                        """
                        if data_dictionary["encoder2_data"] is None:
                            # This is how I use a conditional model, but retain the option of not conditioning on any
                            # image data
                            device = data_dictionary["data"].device
                            batch_size = data_dictionary["data"].shape[0]
                            channels = self.hyper_params["channels"][
                                self.lateral_skip_con_to_use
                            ]
                            spatial_dims = list(data_dictionary["data"].shape[2:])
                            shape_skip = [batch_size, channels] + spatial_dims
                            p_skip = torch.zeros(
                                shape_skip, device=device, dtype=torch.float32
                            )

                            channels = self.hyper_params["channels_per_latent"][
                                self.lateral_skip_con_to_use
                            ]
                            shape = [batch_size, channels] + spatial_dims
                            p_mu = torch.zeros(
                                shape, device=device, dtype=torch.float32
                            )
                            p_log_var = torch.zeros(
                                shape, device=device, dtype=torch.float32
                            )

                            if misc.key_is_true(
                                self.hyper_params, "predict_x_scale_with_sigmoid"
                            ):
                                p_std = torch.ones(
                                    shape, device=device, dtype=torch.float32
                                )

                        else:
                            p_data = data_dictionary["encoder2_data"]

                            # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                            if self.length_of_flag > 0:
                                """
                                Add the flags to p_data, before it goes through convs
                                """
                                flags = data_dictionary["expanded_flag"]
                                if not self.uppermost_block:
                                    if not p_data.shape[2] == flags.shape[2]:
                                        if len(flags.shape) == 4:
                                            flags = flags.repeat(1, 1, 2, 2)
                                        else:
                                            flags = flags.repeat(1, 1, 2, 2, 2)
                                        data_dictionary["expanded_flag"] = flags
                                p_data = torch.cat((p_data, flags), 1)

                            if self.non_imaging_dims > 0:
                                """
                                Add the non_imaging_data to p_data, before it goes through convs
                                """
                                if "non_imaging_data" in data_dictionary:
                                    non_imaging_data = data_dictionary[
                                        "non_imaging_data"
                                    ]
                                else:
                                    batch_size = p_data.shape[0]
                                    device = p_data.device
                                    if len(p_data.shape) == 4:
                                        non_imaging_data = torch.zeros(
                                            (batch_size, self.non_imaging_dims, 1, 1),
                                            device=device,
                                        )
                                    else:
                                        non_imaging_data = torch.zeros(
                                            (
                                                batch_size,
                                                self.non_imaging_dims,
                                                1,
                                                1,
                                                1,
                                            ),
                                            device=device,
                                        )
                                p_data = torch.cat((p_data, non_imaging_data), 1)

                            if self.separate_loc_scale_convs:
                                p_mu = self.activation(p_data)
                                p_mu = self.convs_p_mu[0](p_mu)
                                p_mu = self.activation(p_mu)
                                p_mu = self.convs_p_mu[1](self.pad(p_mu))
                                p_mu = self.activation(p_mu)
                                p_mu = self.convs_p_mu[2](self.pad(p_mu))
                                p_mu = self.activation(p_mu)
                                p_skip = self.convs_p_skip_conv_after_prior(p_mu)
                                p_mu = self.convs_p_mu[3](p_mu)

                                p_log_var = self.activation(p_data)
                                p_log_var = self.convs_p_log_var[0](p_log_var)
                                p_log_var = self.activation(p_log_var)
                                p_log_var = self.convs_p_log_var[1](self.pad(p_log_var))
                                p_log_var = self.activation(p_log_var)
                                p_log_var = self.convs_p_log_var[2](self.pad(p_log_var))
                                p_log_var = self.activation(p_log_var)
                                p_log_var = self.convs_p_log_var[3](p_log_var)
                            else:
                                p_intermediate = self.activation(p_data)
                                p_intermediate = self.convs_p[0](p_intermediate)
                                p_intermediate = self.activation(p_intermediate)
                                p_intermediate = self.convs_p[1](
                                    self.pad(p_intermediate)
                                )
                                p_intermediate = self.activation(p_intermediate)
                                p_intermediate = self.convs_p[2](
                                    self.pad(p_intermediate)
                                )
                                p_intermediate = self.activation(p_intermediate)
                                p_mu = self.convs_p[3](p_intermediate)
                                p_log_var = self.convs_p[4](p_intermediate)
                                p_skip = self.convs_p_skip_conv_after_prior(
                                    p_intermediate
                                )

                            if misc.key_is_true(
                                self.hyper_params, "predict_x_scale_with_sigmoid"
                            ):
                                lower = self.variance_bounds[0]
                                upper = self.variance_bounds[1]
                                p_std = lower + (upper - lower) * torch.sigmoid(
                                    p_log_var
                                )
                            else:
                                if self.variance_bounds is not None:
                                    p_log_var = torch.clamp(
                                        p_log_var,
                                        self.variance_bounds[0],
                                        self.variance_bounds[1],
                                    )

                    q_data = data_dictionary["data"]

                else:
                    q_data = torch.cat(
                        (from_bottom_up_1, data_dictionary["data"]), dim=1
                    )

                    if self.conditional_model:
                        p_data = torch.cat(
                            (from_bottom_up_2, data_dictionary["data"]), dim=1
                        )

                    else:
                        """
                        p_data doesn't use the lateral skips directly, although the incoming 'data' variable
                        *is* a function of the data from the lateral skips by the third latent variable.
                        """
                        p_data = data_dictionary["data"]

                    # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                    if self.length_of_flag > 0:
                        """
                        Add the flags to p_data, before it goes through convs
                        """
                        flags = data_dictionary["expanded_flag"]
                        if not self.uppermost_block:
                            if not p_data.shape[2] == flags.shape[2]:
                                if len(flags.shape) == 4:
                                    flags = flags.repeat(1, 1, 2, 2)
                                else:
                                    flags = flags.repeat(1, 1, 2, 2, 2)
                                data_dictionary["expanded_flag"] = flags
                        p_data = torch.cat((p_data, flags), 1)

                    if self.non_imaging_dims > 0:
                        """
                        Add the non_imaging_data to p_data, before it goes through convs
                        """
                        if "non_imaging_data" in data_dictionary:
                            non_imaging_data = data_dictionary["non_imaging_data"]
                        else:
                            batch_size = p_data.shape[0]
                            device = p_data.device
                            if len(p_data.shape) == 4:
                                non_imaging_data = torch.zeros(
                                    (batch_size, self.non_imaging_dims, 1, 1),
                                    device=device,
                                )
                            else:
                                non_imaging_data = torch.zeros(
                                    (batch_size, self.non_imaging_dims, 1, 1, 1),
                                    device=device,
                                )
                        p_data = torch.cat((p_data, non_imaging_data), 1)

                    if self.separate_loc_scale_convs:
                        p_mu = self.activation(p_data)
                        p_mu = self.convs_p_mu[0](p_mu)
                        p_mu = self.activation(p_mu)
                        p_mu = self.convs_p_mu[1](self.pad(p_mu))
                        p_mu = self.activation(p_mu)
                        p_mu = self.convs_p_mu[2](self.pad(p_mu))
                        p_mu = self.activation(p_mu)
                        p_skip = self.convs_p_skip_conv_after_prior(p_mu)
                        p_mu = self.convs_p_mu[3](p_mu)

                        p_log_var = self.activation(p_data)
                        p_log_var = self.convs_p_log_var[0](p_log_var)
                        p_log_var = self.activation(p_log_var)
                        p_log_var = self.convs_p_log_var[1](self.pad(p_log_var))
                        p_log_var = self.activation(p_log_var)
                        p_log_var = self.convs_p_log_var[2](self.pad(p_log_var))
                        p_log_var = self.activation(p_log_var)
                        p_log_var = self.convs_p_log_var[3](p_log_var)
                    else:
                        p_intermediate = self.activation(p_data)
                        p_intermediate = self.convs_p[0](p_intermediate)
                        p_intermediate = self.activation(p_intermediate)
                        p_intermediate = self.convs_p[1](self.pad(p_intermediate))
                        p_intermediate = self.activation(p_intermediate)
                        p_intermediate = self.convs_p[2](self.pad(p_intermediate))
                        p_intermediate = self.activation(p_intermediate)
                        p_mu = self.convs_p[3](p_intermediate)
                        p_log_var = self.convs_p[4](p_intermediate)
                        p_skip = self.convs_p_skip_conv_after_prior(p_intermediate)

                    if misc.key_is_true(
                        self.hyper_params, "predict_x_scale_with_sigmoid"
                    ):
                        lower = self.variance_bounds[0]
                        upper = self.variance_bounds[1]
                        p_std = lower + (upper - lower) * torch.sigmoid(p_log_var)
                    else:
                        if self.variance_bounds is not None:
                            p_log_var = torch.clamp(
                                p_log_var,
                                self.variance_bounds[0],
                                self.variance_bounds[1],
                            )

                # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                if self.length_of_flag > 0:
                    """
                    Add the flags to q_data, before it goes through convs
                    """
                    flags = data_dictionary["expanded_flag"]
                    if not self.uppermost_block:
                        if not q_data.shape[2] == flags.shape[2]:
                            if len(flags.shape) == 4:
                                flags = flags.repeat(1, 1, 2, 2)
                            else:
                                flags = flags.repeat(1, 1, 2, 2, 2)
                            data_dictionary["expanded_flag"] = flags
                    q_data = torch.cat((q_data, flags), 1)

                if self.non_imaging_dims > 0:
                    """
                    Add the non_imaging_data to p_data, before it goes through convs
                    """
                    if "non_imaging_data" in data_dictionary:
                        non_imaging_data = data_dictionary["non_imaging_data"]
                    else:
                        batch_size = q_data.shape[0]
                        device = q_data.device
                        if len(q_data.shape) == 4:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1), device=device
                            )
                        else:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1, 1),
                                device=device,
                            )
                    q_data = torch.cat((q_data, non_imaging_data), 1)

                if self.separate_loc_scale_convs:
                    q_mu = self.activation(q_data)
                    q_mu = self.convs_q_mu[0](q_mu)
                    q_mu = self.activation(q_mu)
                    q_mu = self.convs_q_mu[1](self.pad(q_mu))
                    q_mu = self.activation(q_mu)
                    q_mu = self.convs_q_mu[2](self.pad(q_mu))
                    q_mu = self.activation(q_mu)
                    q_mu = self.convs_q_mu[3](q_mu)

                    q_log_var = self.activation(q_data)
                    q_log_var = self.convs_q_log_var[0](q_log_var)
                    q_log_var = self.activation(q_log_var)
                    q_log_var = self.convs_q_log_var[1](self.pad(q_log_var))
                    q_log_var = self.activation(q_log_var)
                    q_log_var = self.convs_q_log_var[2](self.pad(q_log_var))
                    q_log_var = self.activation(q_log_var)
                    q_log_var = self.convs_q_log_var[3](q_log_var)
                else:
                    q_intermediate = self.activation(q_data)
                    q_intermediate = self.convs_q[0](q_intermediate)
                    q_intermediate = self.activation(q_intermediate)
                    q_intermediate = self.convs_q[1](self.pad(q_intermediate))
                    q_intermediate = self.activation(q_intermediate)
                    q_intermediate = self.convs_q[2](self.pad(q_intermediate))
                    q_intermediate = self.activation(q_intermediate)
                    q_mu = self.convs_q[3](q_intermediate)
                    q_log_var = self.convs_q[4](q_intermediate)

                if misc.key_is_true(self.hyper_params, "predict_x_scale_with_sigmoid"):
                    lower = self.variance_bounds[0]
                    upper = self.variance_bounds[1]
                    q_std = lower + (upper - lower) * torch.sigmoid(q_log_var)
                else:
                    if misc.key_is_true(
                        self.hyper_params, "model_posterior_as_perturbation_of_prior"
                    ):
                        if not self.uppermost_block:
                            delta_log_var = q_log_var  # For the KL
                            delta_mu = q_mu  # For the KL

                            if "test_op" in self.hyper_params:
                                delta_log_var = torch.clamp(
                                    delta_log_var,
                                    -self.hyper_params["test_op"],
                                    self.hyper_params["test_op"],
                                )
                                delta_mu = torch.clamp(
                                    delta_mu,
                                    -self.hyper_params["test_op"],
                                    self.hyper_params["test_op"],
                                )

                            q_log_var = p_log_var + delta_log_var
                            q_mu = p_mu + delta_mu

                    if self.variance_bounds is not None:
                        q_log_var = torch.clamp(
                            q_log_var, self.variance_bounds[0], self.variance_bounds[1]
                        )

                # if misc.key_is_true(self.hyper_params, 'model_posterior_as_perturbation_of_prior'):
                #     if not self.uppermost_block:
                #         q_log_var = p_log_var + torch.clamp(q_log_var, 0, 0.0001)
                #         q_mu = p_mu + torch.clamp(q_mu, 0, 0.0001)

                # Sample z
                if self.inject_noise:

                    noise = torch.randn_like(q_mu)

                    if "reparam_trick_noise_std" in data_dictionary:
                        noise *= data_dictionary["reparam_trick_noise_std"]

                    if (
                        "noise_injection_multiplier" in self.hyper_params
                        and not self.hyper_params["noise_injection_multiplier"] == 1
                    ):
                        noise *= self.hyper_params["noise_injection_multiplier"]

                    if misc.key_is_true(
                        self.hyper_params, "predict_x_scale_with_sigmoid"
                    ):
                        data = q_mu + torch.mul(q_std, noise)
                    else:
                        data = q_mu + torch.mul(torch.exp(0.5 * q_log_var), noise)
                else:
                    data = q_mu

                # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                if self.length_of_flag > 0:
                    flags = data_dictionary["expanded_flag"]
                    if not self.uppermost_block:
                        if not data.shape[2] == flags.shape[2]:
                            if len(flags.shape) == 4:
                                flags = flags.repeat(1, 1, 2, 2)
                            else:
                                flags = flags.repeat(1, 1, 2, 2, 2)
                            data_dictionary["expanded_flag"] = flags
                    data = torch.cat((data, flags), 1)

                if self.non_imaging_dims > 0:
                    """
                    Add the non_imaging_data to p_data, before it goes through convs
                    """
                    if "non_imaging_data" in data_dictionary:
                        non_imaging_data = data_dictionary["non_imaging_data"]
                    else:
                        batch_size = data.shape[0]
                        device = data.device
                        if len(data.shape) == 4:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1), device=device
                            )
                        else:
                            non_imaging_data = torch.zeros(
                                (batch_size, self.non_imaging_dims, 1, 1, 1),
                                device=device,
                            )
                    data = torch.cat((data, non_imaging_data), 1)

                # Compute the KL
                if not ("veto_KL" in data_dictionary and data_dictionary["veto_KL"]):
                    if "kl_mask" in data_dictionary:
                        kl_res = q_mu.shape[2:]
                        if kl_res == kl_res:
                            kl_mask = data_dictionary["kl_mask"]
                        else:
                            kl_mask = None
                    else:
                        kl_mask = None

                    if misc.key_is_true(
                        self.hyper_params, "predict_x_scale_with_sigmoid"
                    ):
                        if self.uppermost_block and not self.conditional_model:
                            data_dictionary["KL_list"].append(
                                misc.kl_stds(q_mu, q_std, None, None, kl_mask)
                            )
                        else:
                            data_dictionary["KL_list"].append(
                                misc.kl_stds(q_mu, q_std, p_mu, p_std, kl_mask)
                            )
                    else:
                        if self.uppermost_block and not self.conditional_model:
                            data_dictionary["KL_list"].append(
                                misc.kl_log_vars(q_mu, q_log_var, None, None)
                            )
                        else:
                            if misc.key_is_true(
                                self.hyper_params,
                                "model_posterior_as_perturbation_of_prior",
                            ):
                                data_dictionary["KL_list"].append(
                                    misc.kl_perturbed_prior(
                                        delta_mu, delta_log_var, p_log_var
                                    )
                                )
                            else:
                                data_dictionary["KL_list"].append(
                                    misc.kl_log_vars(q_mu, q_log_var, p_mu, p_log_var)
                                )

                # PROGRESSIVE RECONS
                if "res_to_sample_from_prior" in data_dictionary:
                    if data.shape[-1] in data_dictionary["res_to_sample_from_prior"]:
                        """
                        We have found the key that tells us which resolutions to impute using the prior
                        then just recompute them and over-write the sample
                        """
                        data = p_mu

                        # if self.length_of_flag > 0 and 'flag' in data_dictionary:
                        if self.length_of_flag > 0:
                            flags = data_dictionary["expanded_flag"]
                            if not self.uppermost_block:
                                if not data.shape[2] == flags.shape[2]:
                                    # This assumes that all upsampling is x2 in H, W and D
                                    if len(flags.shape) == 4:
                                        flags = flags.repeat(1, 1, 2, 2)
                                    else:
                                        flags = flags.repeat(1, 1, 2, 2, 2)
                                    data_dictionary["expanded_flag"] = flags
                            data = torch.cat((data, flags), 1)

                        if self.non_imaging_dims > 0:
                            """
                            Add the non_imaging_data to p_data, before it goes through convs
                            """
                            if "non_imaging_data" in data_dictionary:
                                non_imaging_data = data_dictionary["non_imaging_data"]
                            else:
                                batch_size = data.shape[0]
                                device = data.device
                                if len(data.shape) == 4:
                                    non_imaging_data = torch.zeros(
                                        (batch_size, self.non_imaging_dims, 1, 1),
                                        device=device,
                                    )
                                else:
                                    non_imaging_data = torch.zeros(
                                        (batch_size, self.non_imaging_dims, 1, 1, 1),
                                        device=device,
                                    )
                            data = torch.cat((data, non_imaging_data), 1)

            """
            This convolution projects self.channels_for_latent channels to self.channels_out channels
            If there's a flag concatenated to the latent, we have adjusted conv_z'z input channels to account for it
            """
            # if not self.channels_for_latent == self.channels_out:
            if "use_rezero" in self.hyper_params and self.hyper_params["use_rezero"]:
                data = self.rezero_alpha * self.conv_z(data)
            else:
                data = self.conv_z(data)

            if self.conditional_model or not self.uppermost_block:
                #  In this case we have a prior, and therefore another residual branch
                if (
                    "use_rezero" in self.hyper_params
                    and self.hyper_params["use_rezero"]
                ):
                    data += self.rezero_alpha * p_skip
                else:
                    data += p_skip

            if self.use_skip_con:
                if self.channels_in == self.channels_out:
                    data += copy_of_incoming_data
                else:
                    data += self.skip_con_conv(copy_of_incoming_data)

            if self.use_batch_norm:
                data = self.batch_norm(data)

            data_dictionary["data"] = data

        return data_dictionary
