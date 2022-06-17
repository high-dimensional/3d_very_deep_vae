import torch.nn as nn
from ..graph_components.conv_block import ConvBlock
from ..graph_components.skip_block import SkipBlock
from ..graph_components.unpooling_block import UnPoolingBlock
from ..graph_components.top_down_block import TopDownBlock
from ..misc import misc
from ..graph_components.tanh_block import TanhBlock
from ..graph_components.sigmoid_block import SigmoidBlock
from ..graph_components.tanh_first_n_channels_block import TanhFirstNChansOnlyBlock
from ..graph_components.sigmoid_first_n_channels_block import (
    SigmoidFirstNChansOnlyBlock,
)


class TopDownGraph:
    """
    This is the top down part of the very deep VAE graph.
    """

    def __init__(self, **kwargs):
        super().__init__()

        hyper_params = kwargs["hyper_params"]

        self.unpool = UnPoolingBlock(
            hyper_params=hyper_params, half_precision=hyper_params["half_precision"]
        )

        if "output_channels" in kwargs:
            output_channels = kwargs["output_channels"]
        else:
            output_channels = 1

        if misc.key_is_true(kwargs, "conditional_prior"):
            self.conditional_prior = True
        else:
            self.conditional_prior = False

        veto_noise_injection = misc.key_is_true(hyper_params, "veto_noise_injection")

        ###########################################################################
        ###########################################################################
        # New code for custom latent sizes
        if "hidden_spatial_dims" in hyper_params:
            self.unpooling_ops = []
            for size in hyper_params["hidden_spatial_dims"][0:-1][::-1]:
                output_size = [size] * 3

                self.unpooling_ops.append(
                    UnPoolingBlock(
                        hyper_params=hyper_params,
                        output_size=output_size,
                        half_precision=hyper_params["half_precision"],
                    )
                )
            output_size = (hyper_params["resolution"],) * 3
            self.unpooling_ops.append(
                UnPoolingBlock(
                    hyper_params=hyper_params,
                    output_size=output_size,
                    half_precision=hyper_params["half_precision"],
                )
            )
        pool_counter = 0
        ###########################################################################
        ###########################################################################

        channels = hyper_params["channels_top_down"][::-1]
        channels_hidden = hyper_params["channels_hidden_top_down"][::-1]
        channels_per_latent = hyper_params["channels_per_latent"][::-1]

        ###############################################################################################
        ###############################################################################################
        self.prior_params = []
        self.prior_params_names = []
        self.posterior_params = []
        self.posterior_params_names = []

        if "latents_to_optimise" in hyper_params:
            self.latents_to_optimise = hyper_params["latents_to_optimise"]
        else:
            self.latents_to_optimise = None

        if "latents_to_use" in hyper_params:
            self.latents_to_use = hyper_params["latents_to_use"]
        else:
            self.latents_to_use = None
        ###############################################################################################
        ###############################################################################################

        # Introducing a counter to give each latent a number starting from 0 at the top
        # It is incremented before each TopDownBlock
        index_of_latent = -1

        latents_per_chanel = hyper_params["latents_per_channel"][::-1]

        self.weight_sharing_index = hyper_params["latents_per_channel_weight_sharing"][
            ::-1
        ]

        groups = []
        self.latents_per_group = []

        # Group 1 (no unpooling)
        channels_in = channels[0]
        channels_hid = channels_hidden[0]
        channels_out = channels[0]

        current_lateral_skip_con = len(channels) - 1

        latents_to_use = latents_per_chanel[0]
        shared_group_ops = None

        hidden_kernel_size = hyper_params["kernel_sizes_top_down"][-1]

        if "length_of_flag" in kwargs:
            length_of_flag = kwargs["length_of_flag"]
            conv_block_channels_in = channels_out + kwargs["length_of_flag"]
            concat_flag = True
        else:
            length_of_flag = 0
            conv_block_channels_in = channels_out
            concat_flag = False

        if "non_imaging_dims" in kwargs:
            non_imaging_dims = kwargs["non_imaging_dims"]
        else:
            non_imaging_dims = 0

        index_of_latent += 1
        latent_counter = 0
        current_block = []
        current_block += [
            TopDownBlock(
                non_imaging_dims=non_imaging_dims,  # Either concat this many zeros to the flat latent, or concat the non-imaging data if it's in the data dictionary
                length_of_flag=length_of_flag,
                conditional_prior=self.conditional_prior,
                channels_in=channels_in,
                channels_hidden=channels_hid,
                channels_out=channels_out,
                channels_for_latent=channels_per_latent[0],
                uppermost_block=True,
                lateral_connection_channels=channels_in,
                lateral_skip_con_to_use=current_lateral_skip_con,
                variance_bounds=hyper_params["variance_hidden_clamp_bounds"],
                precision_reweighting=hyper_params["use_precision_reweighting"],
                separate_loc_scale_convs=hyper_params[
                    "separate_hidden_loc_scale_convs"
                ],
                veto_noise_injection=veto_noise_injection,
                hidden_kernel_size=hidden_kernel_size,
                hyper_params=hyper_params,
                index_of_latent=index_of_latent,
                half_precision=hyper_params["half_precision"],
            )
        ]
        # if self.latents_to_optimise is None or self.latents_to_optimise[-1]:
        #     # Determines whether we optimise the first (top-most) latent
        #     self.prior_params, self.prior_params_names, self.posterior_params, self.posterior_params_names = \
        #         get_convp_and_convq(self.prior_params, self.prior_params_names, self.posterior_params,
        #                             self.posterior_params_names, current_block[-1])
        current_block += [
            ConvBlock(
                channels_in=conv_block_channels_in,
                concat_flag=concat_flag,
                # channels_in=channels_out,
                # channels_hidden=channels_out,
                channels_hidden=channels_hid,
                channels_out=channels_out,
                hidden_kernel_size=hidden_kernel_size,
                hyper_params=hyper_params,
                normalise_weight_by_depth=True,
                half_precision=hyper_params["half_precision"],
            )
        ]
        for n in range(1, latents_to_use):
            latent_counter += 1
            if (
                self.latents_to_use is not None
                and not self.latents_to_use[-1 - latent_counter]
            ):
                # Replace TopDown with ConvBlock
                current_block += [
                    SkipBlock(
                        channels_in=channels_out,
                        channels_out=channels_out,
                        hyper_params=hyper_params,
                        half_precision=hyper_params["half_precision"],
                    )
                ]
            else:
                if "length_of_flag" in kwargs:
                    # No extra condition for adding it to the latent
                    length_of_flag = kwargs["length_of_flag"]
                    conv_block_channels_in = channels_out + kwargs["length_of_flag"]
                    concat_flag = True
                else:
                    length_of_flag = 0
                    conv_block_channels_in = channels_out
                    concat_flag = False

                index_of_latent += 1
                current_block += [
                    TopDownBlock(
                        non_imaging_dims=non_imaging_dims,  # Either concat this many zeros to the flat latent, or concat the non-imaging data if it's in the data dictionary
                        length_of_flag=length_of_flag,  # This was commented out before... Why?
                        conditional_prior=self.conditional_prior,
                        channels_in=channels_out,
                        channels_hidden=channels_hid,
                        channels_out=channels_out,
                        channels_for_latent=channels_per_latent[0],
                        lateral_connection_channels=channels_in,
                        lateral_skip_con_to_use=current_lateral_skip_con,
                        variance_bounds=hyper_params["variance_hidden_clamp_bounds"],
                        precision_reweighting=hyper_params["use_precision_reweighting"],
                        separate_loc_scale_convs=hyper_params[
                            "separate_hidden_loc_scale_convs"
                        ],
                        veto_noise_injection=veto_noise_injection,
                        hidden_kernel_size=hidden_kernel_size,
                        shared_group_ops=shared_group_ops,
                        hyper_params=hyper_params,
                        index_of_latent=index_of_latent,
                        half_precision=hyper_params["half_precision"],
                    )
                ]
                # if self.latents_to_optimise is None or self.latents_to_optimise[-1-latent_counter]:
                #     self.prior_params, self.prior_params_names, self.posterior_params, self.posterior_params_names = \
                #         get_convp_and_convq(self.prior_params, self.prior_params_names, self.posterior_params,
                #                             self.posterior_params_names, current_block[-1])

            current_block += [
                ConvBlock(
                    channels_in=conv_block_channels_in,
                    concat_flag=concat_flag,
                    # channels_in=channels_out,
                    # channels_hidden=channels_out,
                    channels_hidden=channels_hid,
                    channels_out=channels_out,
                    hidden_kernel_size=hidden_kernel_size,
                    normalise_weight_by_depth=True,
                    hyper_params=hyper_params,
                    half_precision=hyper_params["half_precision"],
                )
            ]
            if n == 1 and self.weight_sharing_index[0]:
                print("Weights shared amongst group 0")
                shared_group_ops = current_block[-2]

        groups.append(nn.Sequential(*current_block))
        # groups += current_block

        # Remaining groups
        # for k in range(len(hyper_params['channels']) - 1):
        for k in range(len(channels) - 1):
            channels_in = channels[k]
            channels_hid = channels_hidden[k + 1]
            channels_out = channels[k + 1]
            last_layer = k == len(channels) - 2

            if k == len(channels) - 3:
                channels_out = channels[-2]

            # current_lateral_skip_con = len(hyper_params['channels']) - 2 - k
            # depth = len(hyper_params['channels']) - k - 1
            # current_lateral_skip_con = len(hyper_params['channels_top_down']) - 2 - k

            current_lateral_skip_con = len(channels) - k - 2
            # current_lateral_skip_con = len(channels) - k - 1

            # lateral_connection_channels = hyper_params['channels'][-k-1]
            lateral_connection_channels = hyper_params["channels"][-k - 2]
            latents_to_use = latents_per_chanel[k + 1]
            # latents_to_use = 1
            shared_group_ops = None

            # if k == 0:
            #     # hidden_kernel_size = 2  # Anticipating 2x2 filter maps
            #     hidden_kernel_size = 3  # Anticipating spatial CNN
            # else:
            #     hidden_kernel_size = 3  # Anticipating NxN filter maps, N > 2
            hidden_kernel_size = hyper_params["kernel_sizes_top_down"][-2 - k]

            # if hyper_params['verbose']:
            #     print(f"Top-down block: {k}")

            if latents_to_use == 0:
                if "hidden_spatial_dims" in hyper_params:
                    current_block = [self.unpooling_ops[pool_counter]]
                    pool_counter += 1
                else:
                    current_block = [self.unpool]

                if "length_of_flag" in kwargs:
                    # No extra condition for adding it to the latent
                    conv_block_channels_in = channels_in + kwargs["length_of_flag"]
                    concat_flag = True
                else:
                    conv_block_channels_in = channels_in
                    concat_flag = False

                current_block += [
                    ConvBlock(
                        channels_in=conv_block_channels_in,
                        concat_flag=concat_flag,
                        # channels_in=channels_in,
                        # channels_hidden=channels_out,
                        channels_hidden=channels_hid,
                        channels_out=channels_out,
                        hidden_kernel_size=hidden_kernel_size,
                        veto_bottleneck="do_not_use_bottleneck_in_last_block"
                        in hyper_params
                        and hyper_params["do_not_use_bottleneck_in_last_block"]
                        and last_layer,  # Avoid creating a bottleneck
                        hyper_params=hyper_params,
                        normalise_weight_by_depth=True,
                        veto_skip_connection=last_layer,  # Otherwise 0.5 precision crashes!
                        half_precision=hyper_params["half_precision"],
                    )
                ]
                # current_block += [ConvBlock(channels_in=channels_out,
                #                             # channels_hidden=channels_out,
                #                             channels_hidden=channels_hid,
                #                             channels_out=channels_out,
                #                             hidden_kernel_size=hidden_kernel_size,
                #                             veto_bottleneck=last_layer,  # Avoid creating a bottleneck
                #                             hyper_params=hyper_params,
                #                             half_precision=hyper_params['half_precision'])]
            else:
                latent_counter += 1
                if "hidden_spatial_dims" in hyper_params:
                    current_block = [self.unpooling_ops[pool_counter]]
                    pool_counter += 1
                else:
                    current_block = [self.unpool]
                if (
                    self.latents_to_use is not None
                    and not self.latents_to_use[-1 - latent_counter]
                ):
                    current_block += [
                        SkipBlock(
                            channels_in=channels_in,
                            channels_out=channels_out,
                            hyper_params=hyper_params,
                            half_precision=hyper_params["half_precision"],
                        )
                    ]
                else:
                    index_of_latent += 1

                    if "length_of_flag" in kwargs:
                        length_of_flag = kwargs["length_of_flag"]
                        conv_block_channels_in = channels_out + kwargs["length_of_flag"]
                        concat_flag = True
                    else:
                        length_of_flag = 0
                        conv_block_channels_in = channels_out
                        concat_flag = False

                    current_block += [
                        TopDownBlock(
                            length_of_flag=length_of_flag,
                            conditional_prior=self.conditional_prior,
                            channels_in=channels_in,
                            channels_hidden=channels_hid,
                            channels_out=channels_out,
                            channels_for_latent=channels_per_latent[k + 1],
                            lateral_connection_channels=lateral_connection_channels,
                            lateral_skip_con_to_use=current_lateral_skip_con,
                            variance_bounds=hyper_params[
                                "variance_hidden_clamp_bounds"
                            ],
                            precision_reweighting=hyper_params[
                                "use_precision_reweighting"
                            ],
                            separate_loc_scale_convs=hyper_params[
                                "separate_hidden_loc_scale_convs"
                            ],
                            veto_noise_injection=veto_noise_injection,
                            hyper_params=hyper_params,
                            index_of_latent=index_of_latent,
                            hidden_kernel_size=hidden_kernel_size,
                            half_precision=hyper_params["half_precision"],
                        )
                    ]
                    # if self.latents_to_optimise is None or self.latents_to_optimise[-1 - latent_counter]:
                    #     self.prior_params, self.prior_params_names, self.posterior_params, self.posterior_params_names = \
                    #         get_convp_and_convq(self.prior_params, self.prior_params_names, self.posterior_params,
                    #                             self.posterior_params_names, current_block[-1])

                current_block += [
                    ConvBlock(
                        channels_in=conv_block_channels_in,
                        concat_flag=concat_flag,
                        # channels_in=channels_out,
                        # channels_hidden=channels_out,
                        channels_hidden=channels_hid,
                        channels_out=channels_out,
                        hidden_kernel_size=hidden_kernel_size,
                        normalise_weight_by_depth=True,
                        veto_bottleneck="do_not_use_bottleneck_in_last_block"
                        in hyper_params
                        and hyper_params["do_not_use_bottleneck_in_last_block"]
                        and last_layer,  # Avoid creating a bottleneck
                        hyper_params=hyper_params,
                        veto_skip_connection=last_layer,
                        half_precision=hyper_params["half_precision"],
                    )
                ]
                for n in range(1, latents_to_use):
                    latent_counter += 1
                    if (
                        self.latents_to_use is not None
                        and not self.latents_to_use[-1 - latent_counter]
                    ):
                        current_block += [
                            SkipBlock(
                                channels_in=channels_out,
                                channels_out=channels_out,
                                hyper_params=hyper_params,
                                half_precision=hyper_params["half_precision"],
                            )
                        ]
                    else:
                        index_of_latent += 1

                        if "length_of_flag" in kwargs:
                            conv_block_channels_in = (
                                channels_out + kwargs["length_of_flag"]
                            )
                            concat_flag = True
                        else:
                            conv_block_channels_in = channels_out
                            concat_flag = False

                        current_block += [
                            TopDownBlock(
                                length_of_flag=length_of_flag,  # Specified in block 0
                                conditional_prior=self.conditional_prior,
                                channels_in=channels_out,
                                channels_hidden=channels_hid,
                                channels_out=channels_out,
                                channels_for_latent=channels_per_latent[k + 1],
                                lateral_connection_channels=lateral_connection_channels,
                                lateral_skip_con_to_use=current_lateral_skip_con,
                                variance_bounds=hyper_params[
                                    "variance_hidden_clamp_bounds"
                                ],
                                precision_reweighting=hyper_params[
                                    "use_precision_reweighting"
                                ],
                                separate_loc_scale_convs=hyper_params[
                                    "separate_hidden_loc_scale_convs"
                                ],
                                shared_group_ops=shared_group_ops,
                                veto_noise_injection=veto_noise_injection,
                                hyper_params=hyper_params,
                                index_of_latent=index_of_latent,
                                hidden_kernel_size=hidden_kernel_size,
                                half_precision=hyper_params["half_precision"],
                            )
                        ]
                        # if self.latents_to_optimise is None or self.latents_to_optimise[-1 - latent_counter]:
                        #     self.prior_params, self.prior_params_names, self.posterior_params, self.posterior_params_names = \
                        #         get_convp_and_convq(self.prior_params, self.prior_params_names, self.posterior_params,
                        #                             self.posterior_params_names, current_block[-1])

                    current_block += [
                        ConvBlock(
                            channels_in=conv_block_channels_in,
                            concat_flag=concat_flag,
                            # channels_in=channels_out,
                            channels_hidden=channels_hid,
                            channels_out=channels_out,
                            hidden_kernel_size=hidden_kernel_size,
                            normalise_weight_by_depth=True,
                            veto_bottleneck="do_not_use_bottleneck_in_last_block"
                            in hyper_params
                            and hyper_params["do_not_use_bottleneck_in_last_block"]
                            and last_layer,  # Avoid creating a bottleneck
                            hyper_params=hyper_params,
                            veto_skip_connection=last_layer,
                            half_precision=hyper_params["half_precision"],
                        )
                    ]
                    if n == 1 and self.weight_sharing_index[k + 1]:
                        print("Weights shared amongst group " + str(k + 1))
                        shared_group_ops = current_block[-2]

            groups.append(nn.Sequential(*current_block))

        self.latents = nn.Sequential(*groups).to(kwargs["device"])

        if (
            hyper_params["predict_x_var"]
            and hyper_params["separate_output_loc_scale_convs"]
        ):
            # Need a block to produce mean and log_var of p(x|z).
            # Channel 0 of the output is the mean, 1 is the variance
            block = []
            block += [
                ConvBlock(
                    channels_in=channels[-1],
                    channels_hidden=channels[-1],
                    channels_out=output_channels,
                    veto_bottleneck=True,
                    veto_skip_connection=True,
                    veto_batch_norm=True,
                    output_block_setup=True,
                    hyper_params=hyper_params,
                    half_precision=hyper_params["half_precision"],
                )
            ]

            if "use_tanh_output" in hyper_params and hyper_params["use_tanh_output"]:
                block += [TanhBlock(half_precision=hyper_params["half_precision"])]
            elif (
                "use_sigmoid_output" in hyper_params
                and hyper_params["use_sigmoid_output"]
            ):
                block += [SigmoidBlock(half_precision=hyper_params["half_precision"])]

            self.x_mu = nn.Sequential(*block).to(kwargs["device"])

            block = []
            block += [
                ConvBlock(
                    channels_in=channels[-1],
                    channels_hidden=channels[-1],
                    channels_out=output_channels,
                    veto_bottleneck=True,
                    veto_skip_connection=True,
                    veto_batch_norm=True,
                    output_block_setup=True,
                    hyper_params=hyper_params,
                    half_precision=hyper_params["half_precision"],
                )
            ]

            self.x_var = nn.Sequential(*block).to(kwargs["device"])
        else:
            if hyper_params["predict_x_var"]:
                channels_out = 2 * output_channels
            else:
                channels_out = output_channels

            # Need a block to produce mean and log_var of p(x|z).
            block = []
            block += [
                ConvBlock(
                    channels_in=channels[-1],
                    channels_hidden=channels[-1],
                    channels_out=channels_out,
                    veto_bottleneck=True,
                    veto_skip_connection=True,
                    veto_batch_norm=True,
                    # output_block_setup=True,
                    output_block2_setup=True,
                    hyper_params=hyper_params,
                    half_precision=hyper_params["half_precision"],
                )
            ]

            if (
                "experimental_activate_chans_0_1_only" in hyper_params
                and hyper_params["experimental_activate_chans_0_1_only"]
            ):
                experimental_activate_chans_0_1_only = hyper_params[
                    "experimental_activate_chans_0_1_only"
                ]
            else:
                experimental_activate_chans_0_1_only = False

            if experimental_activate_chans_0_1_only:
                if (
                    "use_tanh_output" in hyper_params
                    and hyper_params["use_tanh_output"]
                ):
                    if hyper_params["predict_x_var"]:
                        out_trans = TanhFirstNChansOnlyBlock(
                            half_precision=hyper_params["half_precision"],
                            channels_to_tanh=output_channels - 1,
                        )
                    else:
                        out_trans = TanhFirstNChansOnlyBlock(
                            half_precision=hyper_params["half_precision"],
                            channels_to_tanh=output_channels - 1,
                        )

                elif (
                    "use_sigmoid_output" in hyper_params
                    and hyper_params["use_sigmoid_output"]
                ):
                    if hyper_params["predict_x_var"]:
                        out_trans = SigmoidFirstNChansOnlyBlock(
                            half_precision=hyper_params["half_precision"],
                            channels_to_sigmoid=output_channels - 1,
                        )
                    else:
                        out_trans = SigmoidFirstNChansOnlyBlock(
                            half_precision=hyper_params["half_precision"],
                            channels_to_sigmoid=output_channels - 1,
                        )
            else:
                if (
                    "use_tanh_output" in hyper_params
                    and hyper_params["use_tanh_output"]
                ):
                    if hyper_params["predict_x_var"]:
                        out_trans = TanhFirstNChansOnlyBlock(
                            half_precision=hyper_params["half_precision"],
                            channels_to_tanh=output_channels,
                        )
                    else:
                        out_trans = TanhBlock(
                            half_precision=hyper_params["half_precision"]
                        )

                elif (
                    "use_sigmoid_output" in hyper_params
                    and hyper_params["use_sigmoid_output"]
                ):
                    if hyper_params["predict_x_var"]:
                        out_trans = SigmoidFirstNChansOnlyBlock(
                            half_precision=hyper_params["half_precision"],
                            channels_to_sigmoid=output_channels,
                        )
                    else:
                        out_trans = SigmoidBlock(
                            half_precision=hyper_params["half_precision"]
                        )

            block += [out_trans]

            self.x_mu = nn.Sequential(*block).to(kwargs["device"])
