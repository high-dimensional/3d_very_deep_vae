import torch.nn as nn
from ..graph_components.conv_block import ConvBlock
from ..graph_components.pooling_block import PoolingBlock
from ..graph_components.unpooling_block import UnPoolingBlock


class BottomUpGraph:
    """
    This is the bottom up part of the very deep VAE graph.
    Each 'ResBlock' is just a 'bottleneck resblock' from the original resnet paper (but with a pair of convs with 3x3
    kernels). They go 1x1xInxA, 3x3xAxB, 3x3xBxC, 1x1xCxD. Then you apply a 1x1xInxD conv to the original input and
    add this to the output.
    -> The depth arg is so I can rescale the output of the block by 1/sqrt(depth), like in the paper
    -> lateral_skip_con_index is used to name the dictionary key so I can find this block's output on the way down
    -> hidden_kernel_size is just so I can turn the 3x3 kernels into 1x1 kernels, for when I'm processing the 1x1
    feature maps in the final bottom-up group of resnet blocks.
    """

    def __init__(self, **kwargs):
        super().__init__()

        hyper_params = kwargs["hyper_params"]
        input_channels = kwargs["input_channels"]

        channels = [input_channels] + hyper_params["channels"]
        channels_hidden = hyper_params["channels_hidden"]
        groups = []

        ###########################################################################
        ###########################################################################
        if "hidden_spatial_dims" in hyper_params:
            # New code for custom latent sizes
            self.pooling_ops = []
            for size in hyper_params["hidden_spatial_dims"]:
                output_size = [size] * 3

                self.pooling_ops.append(
                    UnPoolingBlock(
                        hyper_params=hyper_params,
                        output_size=output_size,
                        half_precision=hyper_params["half_precision"],
                    )
                )
        pool_counter = 0  # HACK. I'm replacing the old /2 pooling with pooling ops from a list that need to be applied
        # in order
        ###########################################################################
        ###########################################################################

        for k in range(len(hyper_params["channels"])):

            if "length_of_flag" in kwargs:
                """
                Initialising the graph with this parameter paves the way to concatenate the flag with *every*
                bottom-up feature map
                """
                channels_in_1 = channels[k] + kwargs["length_of_flag"]
                channels_in_2 = channels[k + 1] + kwargs["length_of_flag"]
                concat_flag = True
            else:
                channels_in_1 = channels[k]
                channels_in_2 = channels[k + 1]
                concat_flag = False

            hidden_kernel_size = hyper_params["kernel_sizes_bottom_up"][k]

            if hyper_params["verbose"]:
                print(f"Bottom-up block: {k}")

            current_block = []
            if (
                "only_use_one_conv_block_at_top" in hyper_params
                and hyper_params["only_use_one_conv_block_at_top"]
                and k == len(hyper_params["channels"]) - 1
            ):
                current_block += [
                    ConvBlock(
                        concat_flag=concat_flag,
                        channels_in=channels_in_1,
                        # channels_in=channels[k],
                        # channels_hidden=channels_hidden[k + 1],
                        channels_hidden=channels_hidden[k],
                        channels_out=channels[k + 1],
                        hidden_kernel_size=hidden_kernel_size,
                        veto_bottleneck=not hyper_params["bottleneck_resnet_encoder"],
                        lateral_skip_con_index=k,  # Adds the block's output to the data dictionary
                        hyper_params=hyper_params,
                        normalise_weight_by_depth=True,
                        half_precision=hyper_params["half_precision"],
                    )
                ]
            else:
                current_block += [
                    ConvBlock(
                        concat_flag=concat_flag,
                        channels_in=channels_in_1,
                        # channels_in=channels[k],
                        # channels_hidden=channels_hidden[k + 1],
                        channels_hidden=channels_hidden[k],
                        channels_out=channels[k + 1],
                        hidden_kernel_size=hidden_kernel_size,
                        veto_bottleneck=not hyper_params["bottleneck_resnet_encoder"],
                        # lateral_skip_con_index=k,  # Adds the block's output to the data dictionary
                        hyper_params=hyper_params,
                        normalise_weight_by_depth=True,
                        half_precision=hyper_params["half_precision"],
                    )
                ]
                current_block += [
                    ConvBlock(
                        concat_flag=concat_flag,
                        channels_in=channels_in_2,
                        # channels_hidden=channels_hidden[k + 1],
                        channels_hidden=channels_hidden[k],
                        channels_out=channels[k + 1],
                        hidden_kernel_size=hidden_kernel_size,
                        veto_bottleneck=not hyper_params["bottleneck_resnet_encoder"],
                        lateral_skip_con_index=k,  # Adds the block's output to the data dictionary
                        hyper_params=hyper_params,
                        normalise_weight_by_depth=True,
                        half_precision=hyper_params["half_precision"],
                    )
                ]

            if k < len(hyper_params["channels"]) - 1:
                # We need a skip con at *every* res (including the lowest, so we can have multiple latent vars at the
                # deepest level, each touching the bottom-up branch).
                if hyper_params["convolutional_downsampling"]:
                    current_block.append(
                        PoolingBlock(
                            hyper_params=hyper_params,
                            channels=channels[k + 1],
                            half_precision=hyper_params["half_precision"],
                        )
                    )
                else:
                    if "hidden_spatial_dims" in hyper_params:
                        current_block.append(self.pooling_ops[pool_counter])
                        pool_counter += 1
                    else:
                        current_block.append(
                            PoolingBlock(
                                hyper_params=hyper_params,
                                half_precision=hyper_params["half_precision"],
                            )
                        )

            groups.append(nn.Sequential(*current_block))

        self.model = nn.Sequential(*groups).to(kwargs["device"])
