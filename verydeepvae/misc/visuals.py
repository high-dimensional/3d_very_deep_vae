import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.axes_grid1 import ImageGrid


def norm_zero_to_one(input, return_min_max_pair=False, input_min_max_pairs=None):
    if np.max(input) == np.min(input):
        output = np.zeros_like(input)
    else:
        if input_min_max_pairs is None:
            the_min = np.min(input)
            the_max = np.max(input)
        else:
            the_min = input_min_max_pairs[0]
            the_max = input_min_max_pairs[1]
        output = (input - the_min) / (the_max - the_min)
    if return_min_max_pair:
        return [output, [the_min, the_max]]
    else:
        return output


def progress_bar(iteration, total, prefix="", suffix="", decimals=2, bar_length=15):
    filled_length = int(round(bar_length * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
    sys.stdout.flush()


def plot_one_slice(
    data_to_plot, title=None, rot90=True, normalise=True, vmin=None, vmax=None
):
    if title is not None:
        plt.title(title)
    plt.axis("off")
    current = np.squeeze(data_to_plot)

    if rot90:
        current = np.rot90(current)

    if normalise:
        current = norm_zero_to_one(current)

    plt.imshow(current, vmin=vmin, vmax=vmax)


def plot_3d_recons_v2(
    data_to_plot,
    titles,
    epoch,
    recon_folder,
    subjects_to_show=5,
    hyper_params=None,
    prefix="",
    postfix="",
    per_subject_prefix=None,
    suppress_progress_bar=False,
    normalise=True,
    vmin=None,
    vmax=None,
    slices_per_axis=3,
    normalise_per_plot_type=None,
    vmin_vmax_per_plot_type=None,
):
    """
    Plot 3 slices per axis for each item in the list 'data_to_plot'
    """
    plot_types = len(data_to_plot)

    subjects_to_show = min(int(data_to_plot[0].shape[0]), subjects_to_show)

    if not isinstance(recon_folder, list):
        recon_folder = [recon_folder] * subjects_to_show

    for i in range(subjects_to_show):
        plt.close("all")
        if epoch:
            plt.suptitle(
                "Epoch: " + str(epoch) + "; subject: " + str(i + 1), fontsize=10
            )
        else:
            plt.suptitle("Subject: " + str(i + 1), fontsize=10)
        fig = plt.gcf()
        fig.set_size_inches(24, 14)
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.gray()

        for j in range(plot_types):
            current = np.squeeze(data_to_plot[j][i])

            if normalise or (
                normalise_per_plot_type is not None and normalise_per_plot_type[j]
            ):
                do_normalise = True
                vmin = 0
                vmax = 1
            else:
                do_normalise = False

            if (
                vmin_vmax_per_plot_type is not None
                and vmin_vmax_per_plot_type[j] is not None
            ):
                current_vmin = vmin_vmax_per_plot_type[j][0]
                current_vmax = vmin_vmax_per_plot_type[j][1]
            elif vmin is not None:
                current_vmin = vmin
                current_vmax = vmax
            else:
                current_vmin = None
                current_vmax = None

            axis_length = current.shape[0]
            axis_length = axis_length // (slices_per_axis + 1)
            for k in range(1, slices_per_axis + 1):
                plt.subplot(
                    plot_types,
                    (3 * slices_per_axis),
                    j * (3 * slices_per_axis) + 0 * slices_per_axis + k,
                )
                plot_one_slice(
                    current[k * axis_length, :, :],
                    normalise=do_normalise,
                    vmin=current_vmin,
                    vmax=current_vmax,
                )

            axis_length = current.shape[1]
            axis_length = axis_length // (slices_per_axis + 1)
            for k in range(1, slices_per_axis + 1):
                if k == np.ceil(slices_per_axis / 2):
                    title = titles[
                        j
                    ]  # Put the title immediately above the middle item!
                else:
                    title = None
                plt.subplot(
                    plot_types,
                    (3 * slices_per_axis),
                    j * (3 * slices_per_axis) + 1 * slices_per_axis + k,
                )
                plot_one_slice(
                    current[:, k * axis_length, :],
                    title=title,
                    normalise=do_normalise,
                    vmin=current_vmin,
                    vmax=current_vmax,
                )

            axis_length = current.shape[2]
            axis_length = axis_length // (slices_per_axis + 1)
            for k in range(1, slices_per_axis + 1):
                plt.subplot(
                    plot_types,
                    (3 * slices_per_axis),
                    j * (3 * slices_per_axis) + 2 * slices_per_axis + k,
                )
                plot_one_slice(
                    current[:, :, k * axis_length],
                    normalise=do_normalise,
                    vmin=current_vmin,
                    vmax=current_vmax,
                )

        if per_subject_prefix:
            plt.savefig(
                os.path.join(
                    recon_folder[i], prefix + per_subject_prefix[i] + postfix + ".png"
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    recon_folder[i],
                    prefix + "_subject_" + str(i + 1) + postfix + ".png",
                )
            )

        if not suppress_progress_bar and subjects_to_show > 1:
            progress_bar(i + 1, subjects_to_show, prefix="Plotting:")


def plot_error_curves(
    data,
    labels,
    plot_title,
    recon_folder,
    postifx="",
    prefix="",
    precision=6,
    xlabel=None,
    ylabel=None,
):
    """
    Plot each curve in the list data (each of which is in [x axis, y axis] format) on
    the same axes.
    """
    plt.close("all")
    plt.figure()

    if xlabel is None:
        plt.xlabel("Epoch")
    else:
        plt.xlabel(xlabel)

    if ylabel is None:
        plt.ylabel("Error")
    else:
        plt.ylabel(ylabel)

    plt.grid()

    # colours = mcolors.CSS4_COLORS
    colours = ["b", "g", "r", "c", "m", "y", "k"] * 8
    line_style = (
        ["-"] * 7
        + ["--"] * 7
        + ["-."] * 7
        + [":"] * 7
        + ["-"] * 7
        + ["--"] * 7
        + ["-."] * 7
        + [":"] * 7
    )

    colours *= 10
    line_style *= 10

    num_plots = len(data)
    for k in range(num_plots):
        current_data = np.stack(data[k])
        current_label = labels[k]
        current_colour = colours[k]
        current_style = line_style[k]

        current_val = round(current_data[-1, 1], precision)
        min_val = round(np.min(current_data[:, 1]), precision)
        appendage = " (current: " + str(current_val) + ", min: " + str(min_val) + ")"

        # current_title = plot_titles[k]
        # plt.plot(current_data[:, 0], current_data[:, 1], 'o-', color=current_colour,
        #          label=current_label + appendage,
        #          ls=current_style)
        plt.plot(
            current_data[:, 0],
            current_data[:, 1],
            color=current_colour,
            label=current_label + appendage,
            ls=current_style,
        )
        # plt.plot(current_data[:, 0], current_data[:, 1], 'o-', color=current_colour,
        #          label=current_label + appendage)

    plt.legend(loc="best")
    plt.suptitle(plot_title, fontsize=10)
    fig = plt.gcf()
    fig.set_size_inches(24, 14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)

    if postifx == "":
        if prefix == "":
            plt.savefig(os.path.join(recon_folder, "error_curves.png"))
        else:
            plt.savefig(os.path.join(recon_folder, prefix + ".png"))
    else:
        if prefix == "":
            plt.savefig(os.path.join(recon_folder, "error_curves_" + postifx + ".png"))
        else:
            plt.savefig(os.path.join(recon_folder, prefix + "_" + postifx + ".png"))


def image_grid(
    data_to_plot,
    epoch,
    recon_folder,
    filename="samples",
    num_to_plot=5,
    norm_recons=False,
):
    """
    Given a numpy array of shape batch x 1 x h x w, this function plots a grid of images
    """

    side_length = int(
        np.max(
            [1, np.min([num_to_plot, np.floor(np.sqrt(0.5 * data_to_plot.shape[0]))])]
        )
    )

    if side_length == 1:
        data_to_plot = data_to_plot[0:1]
    else:
        data_to_plot = data_to_plot[0 : int(2 * side_length * side_length)]

    plt.close("all")
    fig = plt.gcf()
    fig.set_size_inches(24, 14)
    fig.tight_layout()

    if side_length == 1:
        grid = ImageGrid(
            fig=fig, rect=111, nrows_ncols=(side_length, side_length), axes_pad=0.05
        )
    else:
        grid = ImageGrid(
            fig=fig, rect=111, nrows_ncols=(side_length, 2 * side_length), axes_pad=0.05
        )

    if norm_recons:
        vmin = 0
        vmax = 1
    else:
        vmin = vmax = None

    for i, ax in enumerate(grid):
        current = np.squeeze(data_to_plot[i])
        # current = np.squeeze(data_to_plot[i], 0)
        # current = np.squeeze(data_to_plot[i:i+1], 0)  # Complains if there's no length-1 axis...

        if norm_recons:
            current = norm_zero_to_one(current)

        ax.imshow(current, cmap="gray", vmin=vmin, vmax=vmax)

    plt.suptitle("Samples at epoch: " + str(epoch), fontsize=10)
    plt.savefig(os.path.join(recon_folder, filename + ".png"))


def plot_2d(
    data_to_plot,
    titles,
    epoch,
    recon_folder,
    filename="samples",
    is_colour=False,
    num_to_plot=5,
    norm_recons=False,
):
    """
    Given a list of numpy arrays, and corresponding titles, this function plots a
    selection of each
    """
    plt.close("all")
    plot_types = len(data_to_plot)
    counter = 0
    num_to_plot_here = np.min((num_to_plot, data_to_plot[0].shape[0]))

    for k in range(num_to_plot_here):
        counter += 1
        for j in range(plot_types):
            plt.subplot(
                num_to_plot_here,
                plot_types,
                plot_types * counter - (plot_types - j - 1),
            )
            if counter == 1:
                plt.title(titles[j])
            plt.axis("off")
            current = np.squeeze(data_to_plot[j][k])

            if norm_recons:
                current = norm_zero_to_one(current)

            if is_colour:
                current = np.transpose(current, [1, 2, 0])
                plt.imshow(current)
            else:
                if norm_recons:
                    plt.imshow(
                        current, cmap="gray", vmin=0, vmax=1, interpolation="none"
                    )
                else:
                    plt.imshow(current)

    plt.suptitle("Epoch: " + str(epoch), fontsize=10)
    fig = plt.gcf()
    fig.set_size_inches(24, 14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(recon_folder, filename + ".png"))
