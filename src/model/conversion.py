
# > Standard library
import argparse
import logging
import os
import shutil

# > Local dependencies
from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.metrics import CERMetric, WERMetric
from model.optimization import LoghiLearningRateSchedule

# > Third party dependencies
import tensorflow as tf


def convert_model(directory: str,
                  output_directory: str = None,
                  custom_objects: dict = None) -> tf.keras.Model:
    """
    Converts a TensorFlow model to the .keras format for better compatibility.

    Parameters
    ----------
    directory : str
        Directory where the original model is stored.
    output_directory : str
        Directory where the converted model will be saved.
    custom_objects : dict
        Custom objects to be used during model loading.

    Returns
    -------
        tf.keras.Model: The converted TensorFlow model.
    """

    logging.warning("Legacy model format detected. This format will be "
                    "deprecated in the future.")

    try:
        model = tf.keras.models.load_model(
            directory, custom_objects=custom_objects, compile=False)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.error(f"Please look in the directory loghi-htr/utils/convert-v2-to-v3 for instructions on how to convert legacy models to new models.")
        logging.error(f"If using docker use the conversion script in loghi/scripts/convert-v2-to-v3.sh to convert the model.")
        exit(1)

    if output_directory:
        logging.info("Converting model to .keras format...")
        output_folder = os.path.join(output_directory, model.name)

        # Check if there is already a model.keras file in the output folder
        if os.path.exists(os.path.join(output_folder, 'model.keras')):
            logging.warning("A model.keras file already exists in the output "
                            "folder. Attempting to load the model from disk.")
            try:
                return tf.keras.models.load_model(
                    os.path.join(output_folder, 'model.keras'),
                    custom_objects=custom_objects)
            except Exception:
                raise ValueError(
                    "Failed to load the model from disk. Please clean the "
                    f"output folder '{output_folder}' and try again.")

        try:
            # Copy the model directory to the output folder
            shutil.copytree(directory, output_folder)
            shutil.rmtree(os.path.join(output_folder, 'assets'),
                          ignore_errors=True)
            shutil.rmtree(os.path.join(output_folder, 'variables'),
                          ignore_errors=True)

            for file in os.listdir(output_folder):
                if file.endswith('.pb'):
                    os.remove(os.path.join(output_folder, file))
        except Exception as e:
            logging.error(f"Error during file operations: {e}")
            raise

        model.save(os.path.join(output_folder, 'model.keras'))
        logging.info(
            f"Model converted and saved to {output_folder}/model.keras")
    else:
        logging.warning("No output directory specified. The model will not be "
                        "saved to disk.")

    return model


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow model to .keras format.")
    parser.add_argument("--directory", type=str,
                        help="Directory where the original model is stored.")
    parser.add_argument("--output_directory", type=str,
                        help="Directory where the converted model will be "
                        "saved.")

    args = parser.parse_args()

    # Set the custom objects
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    convert_model(args.directory, args.output_directory)
