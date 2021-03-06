# Model configuration

*Schema for JSON file specifying configuration of a very deep variational autoencoder model and training run*

## Properties

- **`total_epochs`** *(integer)*: Total number of epochs to train model for. Exclusive minimum: `0`.
- **`batch_size`** *(integer)*: Number of training points in each minibatch. Exclusive minimum: `0`.
- **`resolution`** *(integer)*: Resolution of (volumetric) images to train model to generate along all dimensions. Must be an integer power of 2. Exclusive minimum: `0`.
- **`channels`** *(array)*: Number of output channels in the encoder's residual network (ResNet) blocks. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`kernel_sizes_bottom_up`** *(array)*: At each resolution (decreasing order), the side lengths of the encoder's kernels. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`kernel_sizes_top_down`** *(array)*: At each resolution (decreasing order), the side lengths of the decoder's kernels. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`channels_hidden`** *(array)*: Number of intermediate channels in the encoder's residual network (ResNet) blocks. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`channels_top_down`** *(array)*: Number of output channels in the decoder's residual network (ResNet) blocks. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`channels_hidden_top_down`** *(array)*: Number of intermediate channels in the decoder's residual network (ResNet) blocks. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`latent_feature_maps_per_resolution`** *(array)*: Number of latent feature maps at each resolution. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`channels_per_latent`** *(array)*: Number of channels per latent feature map. Length must be equal to base-2 logarithm of `resolution` plus one.
  - **Items** *(integer)*: Exclusive minimum: `0`.
- **`random_seed`**: Integer seed to use to initialise state of pseudo-random number generator(s) used in training for reproducible runs. Using a `null` value will result in non-deterministic training runs. Default: `null`.
  - **One of**
    - *null*
    - *integer*: Minimum: `0`. Maximum: `4294967295`.
- **`max_niis_to_use`**: The maximum number of NiFTI files to use in a training epoch. Use this to define a shorter epoch, for example to quickly test visualisations are being saved correctly. Using a `null` value will result in all available files being used. Default: `null`.
  - **One of**
    - *null*
    - *integer*: Exclusive minimum: `0`.
- **`warmup_iterations`** *(integer)*: Iterations to wait before skipping excessively large gradient updates. Exclusive minimum: `0`. Default: `50`.
- **`plot_recons_period`** *(integer)*: Frequency (in epochs) with which to plot reconstructions. Exclusive minimum: `0`. Default: `1`.
- **`subjects_to_plot`** *(integer)*: Number of subjects to include when plotting reconstructions. Exclusive minimum: `0`. Default: `4`.
- **`validation_period`** *(integer)*: Frequency (in epochs) with which to evaluate the model on the validation set. Exclusive minimum: `0`. Default: `1`.
- **`save_period`** *(integer)*: Frequency (in epochs) with which to save checkpoints. Exclusive minimum: `0`. Default: `1`.
- **`l2_reg_coeff`** *(number)*: Coefficient scaling L2 regularization term in objective. Minimum: `0`. Default: `0.0001`.
- **`learning_rate`** *(number)*: Scalar controlling magnitude of stochastic gradient steps. Minimum: `0`. Default: `0.001`.
- **`train_frac`** *(number)*: Fraction of data to use for training with remainder used for validation. Minimum: `0`. Maximum: `1`. Default: `0.95`.
- **`gradient_clipping_value`** *(number)*: Upper limit for the gradient norm, used when clamping gradients before applying gradient updates. Exclusive minimum: `0`. Default: `100.0`.
- **`gradient_skipping_value`** *(number)*: If the gradient norm exceeds this value, skip that iteration's gradient update. Exclusive minimum: `0`. Default: `1000000000000.0`.
- **`scale_hidden_clamp_bounds`** *(array)*: Lower and upper bound on the standard deviation of the prior and posterior Gaussian distributions of the latent variables. Length must be equal to 2. Default: `[0.001, 1]`.
  - **Items** *(number)*: Minimum: `0`.
- **`scale_output_clamp_bounds`** *(array)*: Lower and upper bound on scale (standard deviation of the Gaussian distribution of the input given the latent. Length must be equal to 2. Default: `[0.01, 1]`.
  - **Items** *(number)*: Minimum: `0`.
- **`latent_feature_maps_per_resolution_weight_sharing`**: Either, an array of boolean flags specifying whether to use a shared set of weights to predict the latents at that resolution, one per resolution in decreasing order (that is array length must be equal to base-2 logarithm of `resolution` plus one), or, one of the strings `"all"` or `"none"`, corresponding to flags being `true` or `false` across all resolutions respectively. Default: `"none"`.
  - **One of**
    - *string*: Must be one of: `["none", "all"]`.
    - *array*: Length must be at least 1.
      - **Items** *(boolean)*
- **`latents_to_use`**: Either, an array of boolean flags specifying whether to use each latent feature map (instead of the the deterministic residual network block output), one per latent feature map in order they appear in per-resolution blocks (that is array length must be equal to sum of values in `latent_feature_maps_per_resolution`), or, one of the strings `"all"` or `"none"`, corresponding to flags being `true` or `false` across all feature maps respectively. Default: `"all"`.
  - **One of**
    - *string*: Must be one of: `["none", "all"]`.
    - *array*: Length must be at least 1.
      - **Items** *(boolean)*
- **`latents_to_optimise`**: Either, an array of boolean flags specifying whether to optimise the parameters for the network components controlling each latent feature map, one per latent feature map in order they appear in per-resolution blocks (that is array length must be equal to sum of values in `latent_feature_maps_per_resolution`), or, one of the strings `"all"` or `"none"`, corresponding to flags being `true` or `false` across all feature maps respectively. Default: `"all"`.
  - **One of**
    - *string*: Must be one of: `["none", "all"]`.
    - *array*: Length must be at least 1.
      - **Items** *(boolean)*
- **`half_precision`** *(boolean)*: Whether to train model using 16-bit floating point precision. Default: `false`.
- **`output_activation_function`** *(string)*: Which activation function to use in computing location of Gaussian distribution given latents. Choices are `"tanh"` corresponding to hyperbolic tangent activation function (with range `[-1, 1]`), `"sigmoid"` corresponding to logistic sigmoid (with range `[0, 1]`) or `"identity"` corresponding to identity function. Must be one of: `["tanh", "sigmoid", "identity"]`. Default: `"tanh"`.
- **`plot_gradient_norms`** *(boolean)*: Plot the norms of the gradients after each epoch. Default: `true`.
- **`resume_from_checkpoint`** *(boolean)*: Resume training from a checkpoint. Default: `false`.
- **`restore_optimiser`** *(boolean)*: When resuming training, restore the state of the optimiser (set to false to reset the optimiser's parameters and start training from epoch 1). Default: `true`.
- **`keep_every_checkpoint`** *(boolean)*: Save, and keep, a checkpoint every epoch rather than just keeping the latest one. Default: `true`.
- **`predict_x_scale`** *(boolean)*: Model the scale, not just the location, of the Gaussian distribution of the input given its latent. Default: `true`.
- **`use_precision_reweighting`** *(boolean)*: Re-weight the locations and scales of the prior and posterior distributions of the latents according to the scheme in the paper _Ladder variational autoencoders_ (S??nderby et al. 2016). Default: `false`.
- **`verbose`** *(boolean)*: Print more detail in output during training. Default: `true`.
- **`bottleneck_resnet_encoder`** *(boolean)*: In the encoder, use a three layer Resnet block with a middle layer that has fewer channels than the output layer (the bottleneck). Alternatively, use a two-layer Resnet block whose layers have equal numbers of output channels. Default: `true`.
- **`normalise_weight_by_depth`** *(boolean)*: Normalise each convolution block's randomly initialised kernel parameters by the (square root of the) depth of that block. Default: `true`.
- **`zero_biases`** *(boolean)*: Set each convolution block's bias to zero after initialising it. Default: `true`.
- **`use_rezero`** *(boolean)*: Use skip connections where the 'non-skip' part of the layer is multiplied by a scalar initialised to zero, as described in the paper _ReZero is all you need: fast convergence at large depth_ (Bachlechner et al. 2021). Default: `false`.
- **`veto_batch_norm`** *(boolean)*: Do not use batch normalisation anywhere. Default: `true`.
- **`veto_transformations`** *(boolean)*: Do not apply augmentations to the training data. Default: `false`.
- **`convolutional_downsampling`** *(boolean)*: Down-sample using stride-two convolutions, rather than x2 nearest neighbour downsampling. Default: `false`.
- **`predict_x_scale_with_sigmoid`** *(boolean)*: Predict the scale of the Gaussian distribution of the input given its latent using a (scaled) sigmoid, rather than predicting the natural logarithm of the scale then exponentiating. Default: `true`.
- **`only_use_one_conv_block_at_top`** *(boolean)*: Use a truncated sequence of layers to predict from the latents the location and scale of the Gaussian distribution of the input given its latent. Default: `false`.
- **`separate_hidden_loc_scale_convs`** *(boolean)*: Do not just use one convolutional block, with a two-channel output, to predict the location and scale of the prior and posterior Gaussian distributions of the latents. Instead use separate blocks for the location and scale. Default: `false`.
- **`separate_output_loc_scale_convs`** *(boolean)*: Do not just use one convolutional block, with a two-channel output, to predict the location and scale of the prior and posterior Gaussian distributions of the input given ts latent. Instead use separate blocks for the location and scale. Default: `false`.
- **`apply_augmentations_to_validation_set`** *(boolean)*: Apply to the validation set the same augmentations applied to the training set. Default: `false`.
- **`visualise_training_pipeline_before_starting`** *(boolean)*: Plot examples of the augmented training points before training begins. Default: `true`.
