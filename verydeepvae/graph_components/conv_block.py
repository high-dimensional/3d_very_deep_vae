import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
from ..misc import misc


class ConvBlock(nn.Module):
    """
    The residual block for very deep VAEs minus the 1x1 convs
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.kwargs = kwargs
        self.channels_in = self.kwargs["channels_in"]
        self.channels_hidden = self.kwargs["channels_hidden"]
        self.channels_out = self.kwargs["channels_out"]
        self.hyper_params = kwargs["hyper_params"]
        self.hidden_kernel_size = 3
        self.param_count = 0

        if misc.key_is_true(self.kwargs, 'concat_flag'):
            self.concat_flag = True
        else:
            self.concat_flag = False

        if 'mu_block' in self.kwargs:
            self.mu_block = self.kwargs["normalise_weight_by_depth"]
        else:
            self.mu_block = False

        if 'normalise_weight_by_depth' in self.kwargs:
            self.normalise_weight_by_depth = self.kwargs["normalise_weight_by_depth"]
        else:
            self.normalise_weight_by_depth = False

        if 'depth_override' in self.hyper_params:
            self.depth = self.hyper_params['depth_override']
        else:
            self.depth = np.sum(self.hyper_params['latents_per_channel']) + 2 * (len(self.hyper_params['channels']) - 1)

        if 'veto_bottleneck' in self.kwargs and self.kwargs['veto_bottleneck']:
            self.veto_bottleneck = True
        else:
            # Just use a pair of convolutions in the usual way
            self.veto_bottleneck = False

        if 'hidden_kernel_size' in self.kwargs:
            self.hidden_kernel_size = self.kwargs["hidden_kernel_size"]
            if self.hidden_kernel_size == 1:
                self.pad = torch.nn.Identity()
            elif self.hidden_kernel_size == 2:
                if 'data_is_3d' in self.hyper_params and self.hyper_params['data_is_3d']:
                    self.pad = nn.ConstantPad2d((1, 0, 1, 0, 1, 0), 0)
                else:
                    self.pad = nn.ConstantPad2d((1, 0, 1, 0), 0)
            elif self.hidden_kernel_size == 3:
                if 'data_is_3d' in self.hyper_params and self.hyper_params['data_is_3d']:
                    self.pad = nn.ConstantPad2d((1, 1, 1, 1, 1, 1), 0)
                else:
                    self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)
            else:
                print("Kernels must be 1x or 2x or 3x. Quitting.")
                quit()
        else:
            if 'data_is_3d' in self.hyper_params and self.hyper_params['data_is_3d']:
                self.pad = nn.ConstantPad2d((1, 1, 1, 1, 1, 1), 0)
            else:
                self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)

        if 'data_is_3d' in self.hyper_params and self.hyper_params['data_is_3d']:
            ConvOp = nn.Conv3d
            BatchNormOp = nn.BatchNorm3d
            d = 3
            one_dims = '1x1x1'
            filter_space_dims = str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size) + "x" + \
                                str(self.hidden_kernel_size)
        else:
            ConvOp = nn.Conv2d
            BatchNormOp = nn.BatchNorm2d
            d = 2
            one_dims = '1x1'
            filter_space_dims = str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size)

        if ('veto_batch_norm' in self.kwargs and self.kwargs['veto_batch_norm']) or \
            ('veto_batch_norm' in self.hyper_params and self.hyper_params['veto_batch_norm']):
            self.apply_batch_norm = False
        else:
            self.apply_batch_norm = True
            self.batch_norm = BatchNormOp(num_features=self.channels_out, eps=1e-5, momentum=0.1, affine=True,
                                          track_running_stats=True)
            self.param_count += 2 * self.channels_out

        if 'veto_skip_connection' in self.kwargs and self.kwargs['veto_skip_connection']:
            self.use_skip_connection = False
        else:
            self.use_skip_connection = True

            if not self.channels_in == self.channels_out:
                # if self.channels_out > 3:
                # if not(self.channels_in == 6 and self.channels_out == 3):
                misc.print_0(self.hyper_params, "-> Skip (projection)")
                self.skip_con = ConvOp(in_channels=self.channels_in, out_channels=self.channels_out, kernel_size=1,
                                       stride=1, padding=0, bias=True)
                self.param_count += self.channels_in * self.channels_out

                if 'zero_biases' in self.hyper_params and self.hyper_params['zero_biases']:
                    self.skip_con.bias.data *= 0.0
                # else:
                #     print("")
            else:
                misc.print_0(self.hyper_params, "-> Skip (identity)")

        if 'activation' in self.kwargs:
            if self.kwargs["activation"] is None:
                self.activation = torch.nn.Identity()
            else:
                self.activation = self.kwargs["activation"]
        else:
            self.activation = torch.nn.GELU()

        if 'output_activation' in self.kwargs:
            self.output_activation = self.kwargs['output_activation']
        else:
            self.output_activation = self.activation

        if 'lateral_skip_con_index' in self.kwargs:
            self.lateral_skip_con_index = self.kwargs["lateral_skip_con_index"]
        else:
            self.lateral_skip_con_index = None

        if self.veto_bottleneck:
            if 'output_block_setup' in self.kwargs and self.kwargs['output_block_setup']:
                self.convolutions = nn.ModuleList()
                self.convolutions.append(ConvOp(in_channels=self.channels_in, out_channels=self.channels_in,
                                                kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                                                bias=True))
                self.convolutions.append(ConvOp(in_channels=self.channels_in, out_channels=self.channels_out,
                                                kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                                                bias=True))

                # self.convolutions_bn = nn.ModuleList()
                # self.convolutions_bn.append(
                #     BatchNormOp(num_features=self.channels_in, eps=1e-5, momentum=0.1, affine=True,
                #                 track_running_stats=True))
                # self.convolutions_bn.append(
                #     BatchNormOp(num_features=self.channels_out, eps=1e-5, momentum=0.1, affine=True,
                #                 track_running_stats=True))

                self.param_count += self.hidden_kernel_size ** d * self.channels_in * self.channels_in
                self.param_count += self.hidden_kernel_size ** d * self.channels_in * self.channels_out

                if 'verbose' in self.hyper_params and self.hyper_params['verbose']:
                    out_txt = filter_space_dims + "x" + str(self.channels_in) + "x" + str(self.channels_in)
                    out_txt += ", " + filter_space_dims + "x" + str(self.channels_in) + "x" + str(self.channels_out)
                    print("Conv block kernels (" + str(self.param_count) + "): " + out_txt)
            elif 'output_block2_setup' in self.kwargs and self.kwargs['output_block2_setup']:
                self.convolutions = nn.ModuleList()
                self.convolutions.append(ConvOp(in_channels=self.channels_in, out_channels=self.channels_out,
                                                kernel_size=1, stride=1, padding=0,
                                                bias=True))

                # self.convolutions_bn = nn.ModuleList()
                # self.convolutions_bn.append(
                #     BatchNormOp(num_features=self.channels_in, eps=1e-5, momentum=0.1, affine=True,
                #                 track_running_stats=True))
                # self.convolutions_bn.append(
                #     BatchNormOp(num_features=self.channels_out, eps=1e-5, momentum=0.1, affine=True,
                #                 track_running_stats=True))

                self.param_count += self.channels_in * self.channels_out

                if 'verbose' in self.hyper_params and self.hyper_params['verbose']:
                    out_txt = one_dims + "x" + str(self.channels_in) + "x" + str(self.channels_out)
                    print("Conv block kernels (" + str(self.param_count) + "): " + out_txt)
            else:
                self.convolutions = nn.ModuleList()
                self.convolutions.append(ConvOp(in_channels=self.channels_in, out_channels=self.channels_out,
                                                kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                                                bias=True))
                self.convolutions.append(ConvOp(in_channels=self.channels_out, out_channels=self.channels_out,
                                                kernel_size=self.hidden_kernel_size, stride=1, padding=0,
                                                bias=True))

                # self.convolutions_bn = nn.ModuleList()
                # self.convolutions_bn.append(
                #     BatchNormOp(num_features=self.channels_out, eps=1e-5, momentum=0.1, affine=True,
                #                 track_running_stats=True))
                # self.convolutions_bn.append(
                #     BatchNormOp(num_features=self.channels_out, eps=1e-5, momentum=0.1, affine=True,
                #                 track_running_stats=True))

                self.param_count += self.hidden_kernel_size ** d * self.channels_in * self.channels_out
                self.param_count += self.hidden_kernel_size ** d * self.channels_out * self.channels_out

                if 'verbose' in self.hyper_params and self.hyper_params['verbose']:
                    out_txt = filter_space_dims + "x" + str(self.channels_in) + "x" + str(self.channels_out)
                    out_txt += ", " + filter_space_dims + "x" + str(self.channels_out) + "x" + str(self.channels_out)
                    print("Conv block kernels (" + str(self.param_count) + "): " + out_txt)
        else:
            self.convolutions = nn.ModuleList()
            self.convolutions.append(ConvOp(in_channels=self.channels_in, out_channels=self.channels_hidden,
                                            kernel_size=1, stride=1, padding=0, bias=True))
            self.convolutions.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_hidden,
                                            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
            self.convolutions.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_hidden,
                                            kernel_size=self.hidden_kernel_size, stride=1, padding=0, bias=True))
            self.convolutions.append(ConvOp(in_channels=self.channels_hidden, out_channels=self.channels_out,
                                            kernel_size=1, stride=1, padding=0, bias=True))

            # self.convolutions_bn = nn.ModuleList()
            # self.convolutions_bn.append(
            #     BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
            #                 track_running_stats=True))
            # self.convolutions_bn.append(
            #     BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
            #                 track_running_stats=True))
            # self.convolutions_bn.append(
            #     BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
            #                 track_running_stats=True))
            # self.convolutions_bn.append(
            #     BatchNormOp(num_features=self.channels_hidden, eps=1e-5, momentum=0.1, affine=True,
            #                 track_running_stats=True))

            self.param_count += self.channels_in * self.channels_hidden
            self.param_count += self.hidden_kernel_size ** d * self.channels_hidden * self.channels_hidden
            self.param_count += self.hidden_kernel_size ** d * self.channels_hidden * self.channels_hidden
            self.param_count += self.channels_hidden * self.channels_out

            if 'verbose' in self.hyper_params and self.hyper_params['verbose']:
                out_txt = one_dims + "x" + str(self.channels_in) + "x" + str(self.channels_hidden)
                out_txt += ", " + filter_space_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_hidden)
                out_txt += ", " + filter_space_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_hidden)
                out_txt += ", " + one_dims + "x" + str(self.channels_hidden) + "x" + str(self.channels_out)
                print("Conv block kernels (" + str(self.param_count) + "): " + out_txt)

        if self.use_skip_connection and self.normalise_weight_by_depth:
            self.convolutions[-1].weight.data *= np.sqrt(1 / self.depth)

        if 'zero_biases' in self.hyper_params and self.hyper_params['zero_biases']:
            for a in self.convolutions:
                a.bias.data *= 0.

        if 'use_rezero' in self.hyper_params and self.hyper_params['use_rezero']:
            if self.use_skip_connection:
                self.rezero_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):
            data = data_dictionary['data']

            if self.concat_flag:
                """
                Concatenate the (appropriately expanded) flag to the incoming feature maps
                """
                flags = data_dictionary['flag']
                starting_shape = list(data.shape)
                starting_shape[1] = len(flags)
                posterior_input_channels = torch.zeros(size=starting_shape, dtype=data.dtype,
                                                       device=data.device, requires_grad=False)
                for k in range(len(flags)):
                    posterior_input_channels[:, k, ...] = flags[k]
                data = torch.cat((data, posterior_input_channels), 1)

            copy_of_incoming_data = data.clone()

            if self.veto_bottleneck:
                if 'output_block2_setup' in self.kwargs and self.kwargs['output_block2_setup']:
                    data = self.activation(data)
                    data = self.convolutions[0](data)
                else:
                    data = self.activation(data)
                    data = self.convolutions[0](self.pad(data))
                    data = self.activation(data)
                    data = self.convolutions[1](self.pad(data))

                    if self.apply_batch_norm:
                        data = self.batch_norm(data)

                    # data = self.convolutions[0](self.pad(data))
                    # data = self.convolutions_bn[0](data)
                    # data = self.activation(data)
                    # data = self.convolutions[1](self.pad(data))
                    # data = self.convolutions_bn[1](data)
                    # data = self.output_activation(data)

                    # if self.apply_batch_norm:
                    #     data = self.batch_norm(data)

            else:
                data = self.activation(data)
                data = self.convolutions[0](data)
                data = self.activation(data)
                data = self.convolutions[1](self.pad(data))
                data = self.activation(data)
                data = self.convolutions[2](self.pad(data))
                data = self.activation(data)
                data = self.convolutions[3](data)

                if self.apply_batch_norm:
                    data = self.batch_norm(data)

                # data = self.convolutions[0](data)
                # data = self.convolutions_bn[0](data)
                # data = self.activation(data)
                # data = self.convolutions[1](self.pad(data))
                # data = self.convolutions_bn[1](data)
                # data = self.activation(data)
                # data = self.convolutions[2](self.pad(data))
                # data = self.convolutions_bn[2](data)
                # data = self.activation(data)
                # data = self.convolutions[3](data)
                # data = self.convolutions_bn[3](data)
                # data = self.output_activation(data)

                # if self.apply_batch_norm:
                #     data = self.batch_norm(data)

            if self.use_skip_connection:
                if 'use_rezero' in self.hyper_params and self.hyper_params['use_rezero']:
                    if self.channels_in == self.channels_out:
                        data = self.rezero_alpha * data + copy_of_incoming_data
                    else:
                        data = self.rezero_alpha * data + self.skip_con(copy_of_incoming_data)
                else:
                    if self.channels_in == self.channels_out:
                        data += copy_of_incoming_data
                    else:
                        # if self.channels_out > 3:
                        # if not (self.channels_in == 6 and self.channels_out == 3):
                        data += self.skip_con(copy_of_incoming_data)
                        # else:
                        #     pass
                        # pass
                        # data += self.skip_con(copy_of_incoming_data)

            data_dictionary['data'] = data

            if self.lateral_skip_con_index is not None:
                data_dictionary['lateral_skip_' + str(self.lateral_skip_con_index)] = data

        return data_dictionary
