# Imports

# > Standard Library
import os
import sys
import random
from pathlib import Path

# > Third-party dependencies
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# > Local dependencies
# Add the above directory to the path so it can be used when ran from root dir
sys.path.append(str(Path(__file__).resolve().parents[1] / '../src'))

from vis_arg_parser import get_args  # noqa: E402
from vis_utils import preprocess_image, init_pre_trained_model  # noqa: E402


def visualize_filter(filter_index, channels,
                     feature_extractor, image_width=256):
    """
    Visualize the activation of a specific filter in a convolutional layer
    using gradient ascent.

    Parameters
    ----------
    filter_index : int
        The index of the filter to visualize.
    channels : int
        The number of channels in the input image.
    feature_extractor : tf.keras.Model
        The model or layer used for feature extraction.
    image_width : int, optional
        The width of the input image, by default 256.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - loss : tf.Tensor
            The loss value associated with the filter activation.
        - img : numpy.ndarray
            The visualized image showing the activation of the specified
            filter.

    Examples
    --------
    >>> filter_idx = 0
    >>> num_channels = 3
    >>> resnet50_model = tf.keras.applications.ResNet50(weights='imagenet',
    ... include_top=False)
    >>> loss_value, visualization = visualize_filter(filter_idx, num_channels,
    ... resnet50_model)
    # Returns a tuple with the loss value and the visualized image.
    """
    iterations = 20
    learning_rate = 10
    img = tf.random.uniform((1, image_width, 64, channels))  # Initialize image
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(img)
            activation = feature_extractor(img)
            filter_activation = activation[:, 2:-2, 2:-2, filter_index]
            loss = tf.reduce_mean(filter_activation)
        grads = tape.gradient(loss, img)
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
    img = tf.transpose(img[0].numpy(), perm=[1, 0, 2])
    img = deprocess_image(img)
    return loss, img


def deprocess_image(img):
    """
    Deprocess an image that has been preprocessed for visualization.

    Parameters
    ----------
    img : numpy.ndarray
        The preprocessed image.

    Returns
    -------
    numpy.ndarray
        The deprocessed image suitable for visualization.

    Notes
    -----
    This function reverses the preprocessing steps applied to an image for
    visualization purposes.

    Examples
    --------
    >>> deprocessed_img = deprocess_image(img)
    # Returns the deprocessed image for visualization.
    """
    img /= 2.0
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def select_number_of_row_plots(model, do_detailed=False):
    """
    Select layers for plotting based on the number of filters.

    Parameters
    ----------
    model : tf.keras.Model
        The model for which layers need to be selected.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - layer_info : list
            Information about selected layers, including name, kernel size, and
            filters.
        - sub_list_indices : list
            Indices of selected layers in the original layer list.
        - conv_layers : list
            Keras conv layers taken directly from the model

    Notes
    -----
    This function analyzes the layers of a given model and selects layers for
    plotting. If detailed analysis is requested, it returns information for all
    convolutional layers; otherwise, it selects layers based on unique filter
    values.

    Examples
    --------
    >>> model = tf.keras.applications.ResNet50(weights='imagenet',
    ... include_top=False)
    >>> layer_info, indices = select_number_of_row_plots(model, True)
    # Returns a tuple with information about selected layers and their indices.
    """
    layer_info = [{"layer_name": layer.name.lower(),
                   "kernel_size": str(layer.kernel_size),
                   "filters": str(layer.filters)}
                  for layer in model.layers if
                  layer.name.lower().startswith("conv")]
    conv_layers = [layer for layer in model.layers
                   if layer.name.lower().startswith("conv")]
    if not do_detailed:
        filter_occurrence_dict = {}
        sub_list_indices = []
        layer_selection = []
        for i, layer_data in enumerate(layer_info):
            filter_value = layer_data.get("filters")
            if filter_value not in filter_occurrence_dict:
                filter_occurrence_dict[filter_value] = i
                sub_list_indices.append(i)
                layer_selection.append(layer_info[i])
        return layer_selection, sub_list_indices, conv_layers
    return layer_info, [], conv_layers


def main(args=None):
    # Load args
    if args is None:
        args = get_args()

    # Load in pre-trained model and get model channels
    model, model_channels, _, model_height = init_pre_trained_model()

    # Prep plots
    num_filters_per_row = args.num_filters_per_row  # Number of filters per row

    # Get the conv layer info, indices in layer list and the actual conv layers
    layer_info, sub_list_indices, conv_layers = select_number_of_row_plots(
        model)

    # Top level plot
    if args.dark_mode:
        plt.style.use('dark_background')
    fig = plt.figure(figsize=(5 * num_filters_per_row,
                     num_filters_per_row), dpi=200)
    sub_figs = fig.subfigures(len(layer_info), ncols=1)

    # Get layout parameters for later plots
    fig.tight_layout()
    dic = {par: getattr(fig.subplotpars, par)
           for par in ["left", "right", "bottom", "wspace"]}

    # Set base row_index
    row_index = 0

    # Select relevant layers
    layer_list = conv_layers if len(sub_list_indices) == 0 else [
        conv_layers[i] for i in sub_list_indices]

    for layer in layer_list:
        print("Plotting filters for layer: ", layer_info[row_index])
        feature_extractor = tf.keras.Model(
            inputs=model.inputs, outputs=layer.output)

        # Set subplot layout
        num_subplot_rows = 2 if args.sample_image_path else 1
        filter_plot = sub_figs[row_index].subplots(
            num_subplot_rows, num_filters_per_row)
        filter_cmap = 'viridis' if model_channels > 1 else 'gray'

        selected_filters = int(layer_info[row_index].get("filters"))

        # Randomly select filter indices
        if num_filters_per_row > selected_filters:
            raise ValueError("The num_filters_per_row is higher than the "
                             "number of filters in the conv layer, please "
                             "select a lower num_filters_per_row")
        random_filter_indices = random.sample(range(selected_filters),
                                              num_filters_per_row)
        filter_images = []
        feature_maps = []
        for filter_index in range(num_filters_per_row):
            try:
                if args.sample_image_path:
                    if not os.path.exists(args.sample_image_path):
                        raise FileNotFoundError("Please provide a valid path "
                                                "to a sample image, you "
                                                "provided: "
                                                + args.sample_image_path)
                    img_path = args.sample_image_path

                    # Prepare image based on model channels
                    image, image_width, _ = preprocess_image(
                        img_path, model_channels, model_height)
                    maps = feature_extractor.predict(image)

                    # Add the filter images
                    _, image = visualize_filter(
                        random_filter_indices[filter_index],
                        model_channels,
                        feature_extractor,
                        image_width
                    )
                    filter_images.append(image)

                    # Add the feature maps
                    feature_maps.append(
                        maps[0, :, :, random_filter_indices[filter_index]].T)

                    # If filter_plot has 2 rows then fill each row else do the
                    # regular one
                    filter_plot[0, filter_index].imshow(
                        filter_images[filter_index], cmap=filter_cmap)
                    filter_plot[0, filter_index].set_title(
                        "Conv Filter: " + str(filter_index),
                        fontdict={'fontsize': 10},
                        loc='left')
                    filter_plot[1, filter_index].imshow(
                        feature_maps[filter_index], cmap='gray')
                    filter_plot[1, filter_index].set_title(
                        "Layer Activations: " + str(filter_index),
                        fontdict={'fontsize': 10},
                        loc='left')
                else:
                    # Add the filter images
                    _, image = visualize_filter(
                        random_filter_indices[filter_index],
                        feature_extractor,
                        model_channels
                    )
                    filter_images.append(image)

                    # Individual plot level
                    filter_plot[filter_index].imshow(
                        filter_images[filter_index], cmap=filter_cmap)
                    filter_plot[filter_index].set_title(
                        "Conv Filter: " + str(filter_index))
            except IndexError as e:
                raise IndexError("filter_index has surpassed the filters in "
                                 "the layer, select a lower number of filters "
                                 "to plot") from e

        # Fix layout parameters and keep some extra space at the top for
        # suptitle
        fig.subplots_adjust(**dic, top=0.7, hspace=0.5)

        # Disable axes
        for ax in filter_plot.ravel():
            ax.set_axis_off()

        # Display the layer_name above the subplots
        layer_info_dict = layer_info[row_index]
        sub_figs[row_index].suptitle(f"{layer_info_dict.get('layer_name')}: "
                                     "kernel_size: "
                                     f"{layer_info_dict.get('kernel_size')}: "
                                     "filters: "
                                     f"{layer_info_dict.get('filters')}",
                                     fontsize=12,
                                     weight='bold',
                                     ha='left',
                                     x=0.125)
        row_index += 1

        os.makedirs(Path(__file__).with_name("visualize_plots"), exist_ok=True)

        output_name = (args.existing_model.split("/")[-2]
                       + ("_filters_act" if args.sample_image_path
                          else "_filters")
                       + ("_dark" if args.dark_mode
                          else "_light")
                       + ("_detailed.png" if args.do_detailed
                          else ".png"))
        plt.savefig(os.path.join(str(Path(__file__)
                                     .with_name("visualize_plots"))
                                 + "/" + output_name),
                    bbox_inches='tight')


if __name__ == '__main__':
    main()
