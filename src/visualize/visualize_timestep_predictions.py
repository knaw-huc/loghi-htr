# Imports

# > Standard Library
import os
from pathlib import Path
import sys
import xlsxwriter
import re
from typing import Tuple, List

# > Third party libraries
import numpy as np
import cv2

import vis_utils

# Prep GPU support and set seeds/objects
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '0'

# Import tensorflow after setting the environment variables
import tensorflow as tf  # noqa: E402

# > Local dependencies
# Add the above directory to the path
sys.path.append(str(Path(__file__).resolve().parents[1] / '../src'))
from vis_arg_parser import get_args  # noqa: E402
from vis_utils import preprocess_image, init_pre_trained_model  # noqa: E402


def remove_tags(text: str) -> str:
    """
    Remove special tags from the input text.

    Parameters
    ----------
    text : str
        The input text containing special tags to be removed.

    Returns
    -------
    str
        The text with special tags removed.

    Notes
    -----
    This function removes specific Unicode characters representing tags:
    - ␃ (U+2403): STRIKETHROUGHCHAR
    - ␅ (U+2405): UNDERLINECHAR
    - ␄ (U+2404): SUBSCRIPTCHAR
    - ␆ (U+2406): SUPERSCRIPTCHAR

    Examples
    --------
    >>> remove_tags("Hello␃ World␅!")
    'Hello World!'
    >>> remove_tags("2␄ + 3␆ = 5")
    '2 + 3 = 5'
    """
    # public static String STRIKETHROUGHCHAR = "␃";
    text = text.replace('␃', '')
    # public static String UNDERLINECHAR = "␅";
    text = text.replace('␅', '')
    # public static String SUBSCRIPTCHAR = "␄";
    text = text.replace('␄', '')
    # public static String SUPERSCRIPTCHAR = "␆";
    text = text.replace('␆', '')
    return text


def find_blanks(text_line, char_list):
    blanks = 0
    for step in text_line:
        char_index = tf.get_static_value(tf.math.argmax(step))
        if char_index == len(char_list):
            blanks += 1
    return blanks


def get_timestep_indices(model_path: str, preds: np.ndarray,
                         image_width: tf.Tensor) -> Tuple[str,
                                                          List[int],
                                                          List[List[int]],
                                                          float,
                                                          float]:
    """
    Retrieve timestep indices and related information from a model.

    Parameters
    ----------
    model_path : str
        The path to the model.
    preds : numpy.ndarray
        Predictions from the model.
    image_width : tf.Tensor
        The width of the preprocessed image.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - char_list : str
            The content of the 'charlist.txt' file.
        - timestep_char_list_indices : list
            List of indices representing the argmax of each timestep.
        - timestep_char_list_indices_top_3 : list
            List of indices representing the top k values of each timestep.
        - step_width : float
            The width of each timestep calculated based on the model and image
            width.
        - pad_steps_skip : float
            The number of steps to skip for padding, calculated based on step
            width.

    Notes
    -----
    This function reads 'charlist.txt' from the provided model path, extracts
    information related to timesteps, and calculates the step width and padding
    steps.

    Examples
    --------
    >>> model_path = "/path/to/your/model/"
    >>> char_list, indices, top_3_indices, width, pad_skip = \
    ...     get_timestep_indices(model_path, preds, image_width)
    >>> print(char_list)
    'abcde...'
    >>> print(indices)
    [0, 3, 1, ...]
    >>> print(top_3_indices)
    [[0, 1, 2], [2, 0, 4], ...]
    >>> print(width)
    10.5
    >>> print(pad_skip)
    4.0
    """
    char_list = model_path + "charlist.txt"
    with open(char_list, 'r', encoding="utf-8") as f:
        char_list = f.read()

    timestep_char_list_indices = []
    timestep_char_list_indices_top_3 = []
    top_k = 3
    step_width = 0
    pad_steps_skip = 0
    for text_line in preds:
        timesteps = len(text_line)
        blanks = find_blanks(text_line, char_list)
        timesteps = timesteps - blanks
        print("timesteps: ", timesteps)
        step_width = (tf.get_static_value(image_width) + vis_utils.width_padding) / timesteps
        print("step_width: ", step_width)
        pad_steps_skip = vis_utils.width_padding // step_width
        print("pad_steps_skip: ", pad_steps_skip)
        for time_step in text_line:
            timestep_char_list_indices.append(tf.get_static_value(tf.math.argmax(time_step)))
            timestep_char_list_indices_top_3.append(tf.get_static_value(tf.math.top_k(time_step, k=top_k, sorted=True)))
    return char_list, timestep_char_list_indices, \
        timestep_char_list_indices_top_3, step_width, pad_steps_skip


def write_ctc_table_to_xlsx(preds: np.ndarray,
                           char_list: str,
                           index_correction: int) -> None:
    """
    Write CTC (Connectionist Temporal Classification) table data to a xlsx file.

    Parameters
    ----------
    preds : numpy.ndarray
        3D array representing the predictions from the CTC model.
    char_list : str
        String containing the characters used in the predictions.
    index_correction : int
        Correction factor for indexing characters in the output.

    Notes
    -----
    This function takes CTC predictions in the form of a 3D array, extracts the
    data, and writes it to a xlsx file. It creates columns for each timestep and
    includes character labels. Conditional formatting is added to better
    distinguish between low and high prediction probabilities.

    Examples
    --------
    >>> preds = np.array([[[0.1, 0.2, 0.3],
    ...                    [0.4, 0.5, 0.6]],
    ...                   [[0.7, 0.8, 0.9],
    ...                    [0.2, 0.1, 0.3]]])
    >>> char_list = "ABC"
    >>> index_correction = 1
    >>> write_ctc_table_to_xlsx(preds, char_list, index_correction)
    # Creates a xlsx file with characters and corresponding predictions for each
    timestep.
    """
    # Iterate through each index and tensor in preds
    tensor_data = []
    for tensor in preds:
        # Iterate through each time step in the tensor
        for time_step in tensor:
            # Add the time step to the row
            tensor_data.append(np.round(time_step,3).tolist())

    # Create columns
    columns = ["ts_" + str(i) for i in range(preds.shape[1])]
    additional_chars = ['MASK', 'BLANK'] if '' in char_list else ["BLANK"]

    # Adjust char_list for special characters
    characters = list(char_list) + additional_chars
    transposed_data = np.transpose(tensor_data)

    # Ensure the visualize_plots directory exists
    directory_path = Path(__file__).with_name("visualize_plots")
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)

    # Setup the XLSX file
    workbook = xlsxwriter.Workbook(str(directory_path / "sample_image_preds.xlsx"))
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})

    # Write the header
    for i, column in enumerate(['Chars'] + columns):
        worksheet.write(0, i, column, bold)

    # Write the rows with index and data
    for i, row in enumerate(transposed_data):
        if i + index_correction > -1:
            # Replace "unseeable" characters for readability in XLSX
            char = (characters[i + index_correction].replace(" ","SPACE")
                                                    .replace("\t", "TAB")
                                                    .replace("\n", "NEWLINE"))
            worksheet.write(i + 1, 0, char, bold)
            for j, cell in enumerate(row):
                worksheet.write(i + 1, j + 1, cell)

    # Apply conditional formatting to create a heatmap effect
    # Adjust the range accordingly to your data (+1 due to header row)
    worksheet.conditional_format(1, 1, len(transposed_data) + 1, len(columns), {
        'type': '2_color_scale',
        'min_color': "#FFFFFF",  # White
        'max_color': "#00FF00"   # Green
    })

    workbook.close()


def create_timestep_plots(bordered_img: np.ndarray, index_correction: int,
                          font_color: Tuple[int, int, int], step_width: int,
                          pad_steps_skip: int, image_height: tf.Tensor,
                          char_list: List[str],
                          timestep_char_list_indices: List[int],
                          timestep_char_list_indices_top_3: List[tf.Tensor]) \
        -> None:
    """
    Create plots with time-step predictions for the provided image.

    Parameters
    ----------
    bordered_img : numpy.ndarray
        The image on which the time-step predictions will be plotted.
    index_correction : int
        Correction factor for indexing characters in the output.
    font_color : tuple
        Tuple representing the color of the font.
    step_width : int
        Width of each time-step prediction in pixels.
    pad_steps_skip : int
        Number of initial time steps to skip for better readability.
    image_height : tf.Tensor
        Height of the input image.
    char_list : List[str]
        List of characters used in the predictions.
    timestep_char_list_indices : List[int]
        List of character indices for each time step.
    timestep_char_list_indices_top_3 : List[tf.Tensor]
        List of tensors containing the top-5 character indices for each time
        step.

    Notes
    -----
    This function adds time-step predictions, including the top-1 prediction
    and top-2 to top-5 most probable characters, to the provided image. It also
    includes additional information such as the final result.

    Examples
    --------
    >>> bordered_img = np.zeros((500, 500, 3), dtype=np.uint8)
    >>> index_correction = 1
    >>> font_color = (255, 255, 255)
    >>> step_width = 20
    >>> pad_steps_skip = 2
    >>> image_height = 300
    >>> char_list = ["a", "b", "c"]
    >>> timestep_char_list_indices = [0, 1, 2]
    >>> timestep_char_list_indices_top_3 = [tf.constant([[0, 1, 2, 3, 4]]),
    ...                                     tf.constant([[1, 2, 0, 3, 4]]),
    ...                                     tf.constant([[2, 1, 0, 3, 4]])]
    >>> create_timestep_plots(bordered_img, index_correction, font_color,
    ...                       step_width, pad_steps_skip, image_height,
    ...                       char_list, timestep_char_list_indices,
    ...                       timestep_char_list_indices_top_3)
    # Updates the provided image with time-step predictions.
    """
    cv2.putText(bordered_img, "Time-step predictions (Top-1 prediction):",
                org=(0, 15),
                color=font_color,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                lineType=cv2.LINE_AA,
                fontScale=0.5,
                thickness=1)

    timestep_char_labels_cleaned = []

    line_start = 0
    # Retrieve the top 5 predictions for each timestep and write them down
    # underneath each other
    last_character = ""
    character_list_length = len(char_list)
    for index, char_index in enumerate(timestep_char_list_indices):
        line_start += step_width
        start_point = (round(line_start), 50)
        end_point = (round(line_start), tf.get_static_value(image_height) + 50)
        # char_list + 1 is blank token, which is final character

        blank_character = char_index == character_list_length+1
        if blank_character:
            character = "BLANK"
            last_character = character
            continue
        character = remove_tags(char_list[char_index + index_correction])
        if last_character == character:
            continue
        last_character = character
        timestep_char_labels_cleaned.append(character)
        if index < pad_steps_skip:
            continue
            # Do not draw predictions that are spaces or
            # "invisible characters" for better readability
        if re.sub(r'\W+', ' ', character).strip()\
                in [" ", ""]:
            # Do not increment the line_start again
            continue

        cv2.line(bordered_img,
                 start_point,
                 end_point,
                 color=(0, 0, 255),
                 thickness=1)
        cv2.putText(bordered_img,
                    character,
                    org=start_point,
                    color=(0, 0, 255),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    lineType=cv2.LINE_AA,
                    fontScale=1,
                    thickness=1)

        # Add the top-2 to top-5 most probable characters for the current
        # time step
        table_start_height = 170
        for top_char_index in \
                timestep_char_list_indices_top_3[index].indices.numpy()[1:]:
            if top_char_index < len(char_list) + 1:
                cv2.putText(bordered_img,
                            remove_tags(char_list[top_char_index +
                                                  index_correction]),
                            org=(round(line_start), table_start_height),
                            color=(0, 0, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            lineType=cv2.LINE_AA,
                            fontScale=1,
                            thickness=1)
                table_start_height += 50

    # Add the top-5 for the specific timestep
    row_num = 2
    # 3 stands for the top-k most probable characters defined in the
    max_row_y = 170 + ((3 - 1) * 50)
    for row_height in range(170, max_row_y, 50):
        # Draw the top-5 text
        cv2.putText(bordered_img, "Top-" + str(row_num),
                    org=(0, row_height),
                    color=font_color,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    lineType=cv2.LINE_AA,
                    fontScale=0.5,
                    thickness=1)
        row_num += 1

    # Add misc. text
    cv2.putText(bordered_img,
                "Other predictions for this time step (lower probability):",
                org=(0, 140),
                color=font_color,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                lineType=cv2.LINE_AA,
                fontScale=0.5,
                thickness=1)
    cv2.putText(bordered_img,
                "Final result (collapsed blank characters):",
                org=(0, 270),
                color=font_color,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                lineType=cv2.LINE_AA,
                fontScale=0.5,
                thickness=1)

    #Initial font_scale
    font_scale = 1

    # Initial estimate of text size
    text_size = cv2.getTextSize("".join(timestep_char_labels_cleaned),
                                cv2.FONT_HERSHEY_DUPLEX,
                                font_scale,
                                1)[0]

    # Adjust font_scale until text width is less than image width
    while text_size[0] > bordered_img.shape[1]:
        # Decrease font_scale by a small amount
        font_scale -= 0.1
        text_size = cv2.getTextSize("".join(timestep_char_labels_cleaned),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    font_scale,
                                    1)[0]

    # Add the cleaned version of the prediction with scaled font
    cv2.putText(bordered_img,
                "".join(timestep_char_labels_cleaned),
                org=(0, 300),
                color=font_color,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                lineType=cv2.LINE_AA,
                fontScale=font_scale,
                thickness=1)
    print("".join(timestep_char_labels_cleaned))


def main(args=None):
    # Load args
    if args is None:
        args = get_args()

    # Load in pre-trained model and get model channels
    model, model_channels, model_path, model_height = init_pre_trained_model()

    # Make sure image is provided with call
    if args.sample_image_path:
        if not os.path.exists(args.sample_image_path):
            raise FileNotFoundError("Please provide a valid path to a sample "
                                    "image, you provided: "
                                    + args.sample_image_path)
        image_path = args.sample_image_path
    else:
        raise ValueError("Please provide a path to a sample image")

    # Prepare image based on model channels
    image, image_width, image_height = preprocess_image(image_path, model_channels, model_height)
    predictions = model.predict(image)

    # Read char_list and calculate character indices for sample image preds
    (char_list, timestep_char_list_indices, timestep_char_list_indices_top_3,
     step_width, pad_steps_skip) = get_timestep_indices(model_path, predictions,
                                                        image_width)

    # Take the "raw" sample image and plot the pred results on top
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original_image_padded = cv2.resize(
        original_image, (tf.get_static_value(image_width) + vis_utils.width_padding, 64))

    # Set index_correction in case masking was used
    # if blank token in char_list -> take normal else i-1
    if '' in char_list:
        index_correction = 0
    else:
        index_correction = -1

    # Set color_scheme
    if args.dark_mode:
        background_color, font_color = [0, 0, 0], (255, 255, 255)  # Dark mode
    else:
        background_color, font_color = [255, 255, 255], (0, 0, 0)  # Light mode

    # Dynamically calculate the right pad width required to keep all text
    # readable (465px width at least)
    additional_right_pad_width = 465 - original_image_padded.shape[1] \
        if original_image_padded.shape[1] < 465 \
        else 50
    bordered_img = cv2.copyMakeBorder(original_image_padded,
                                      top=50,
                                      bottom=200,
                                      left=0,
                                      right=additional_right_pad_width,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=background_color)

    # Create time_step plots
    create_timestep_plots(bordered_img, index_correction, font_color,
                          step_width, pad_steps_skip, image_height,
                          char_list, timestep_char_list_indices,
                          timestep_char_list_indices_top_3)

    # Take character preds for sample image and create xlsx file
    write_ctc_table_to_xlsx(predictions, char_list, index_correction)

    # Double size
    new_width = bordered_img.shape[1] * 2
    new_height = bordered_img.shape[0] * 2

    # Resize the image to double its size and interpolate
    bordered_img_resized = cv2.resize(bordered_img, (new_width, new_height),
                                      interpolation=cv2.INTER_CUBIC)

    # Save the timestep plot
    cv2.imwrite(str(Path(__file__).with_name("visualize_plots"))
                + "/timestep_prediction_plot"
                + ("_dark" if args.dark_mode else "_light")
                + ".jpg",
                bordered_img_resized)


if __name__ == "__main__":
    main()
