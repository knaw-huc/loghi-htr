# Imports

# > Standard library
import logging
import os
import random

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# > Local dependencies
from setup.config import Config


class TensorFlowLogFilter(logging.Filter):
    """Filter to exclude specific TensorFlow logging messages.

    This filter checks each log record for specific phrases that are to be
    excluded from the logs. If any of the specified phrases are found in a log
    message, the message is excluded from the logs.
    """

    def filter(self, record):
        # Exclude logs containing the specific message
        exclude_phrases = [
            "Reduce to /job:localhost/replica:0/task:0/device:CPU:",
        ]
        return not any(phrase in record.msg for phrase in exclude_phrases)


def set_deterministic(seed: int) -> None:
    """
    Sets the environment and random seeds to ensure deterministic behavior in
    TensorFlow operations.

    Parameters
    ----------
    seed : int
        The seed value to be used for setting deterministic operations across
        various libraries.

    Notes
    -----
    This function configures the environment to enforce deterministic behavior
    in TensorFlow by setting the 'TF_DETERMINISTIC_OPS' environment variable.
    It also initializes the seeds for Python's `random`, NumPy's `np.random`,
    and TensorFlow's `tf.random` with the specified seed value. This setup
    is useful for ensuring reproducibility in experiments.
    """

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_environment(config: Config) -> tf.distribute.Strategy:
    """
    Sets up the environment for running the TensorFlow model, including GPU
    configuration and distribution strategy.

    Parameters
    ----------
    config : Config
        The configuration object containing the parsed arguments.

    Returns
    -------
    tf.distribute.Strategy
        The TensorFlow distribution strategy based on the provided arguments.

    Notes
    -----
    This function configures the visible GPU devices based on the 'gpu'
    argument and initializes a TensorFlow distribution strategy. It also logs
    the GPU devices being used and the precision policy (float32 or
    mixed_float16) based on the 'use_float32' and 'gpu' arguments.
    """

    # Initial setup
    logging.info("Running with config:\n%s", config)

    # Set the random seed
    if config["deterministic"]:
        set_deterministic(config["seed"])

    # Set the GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    logging.info("Selected GPU indices from config: %s", config["gpu"])
    logging.info("Available GPU(s): %s", gpu_devices)

    # Set the active GPUs depending on the 'gpu' argument
    if config["gpu"] == "-1":
        active_gpus = []
    elif config["gpu"].lower() == "all":
        active_gpus = gpu_devices
    else:
        gpus = config["gpu"].split(',')
        active_gpus = []
        for i, gpu in enumerate(gpu_devices):
            if str(i) in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                active_gpus.append(gpu)

    if active_gpus:
        logging.info("Using GPU(s): %s", active_gpus)
    else:
        logging.info("Using CPU")

    tf.config.set_visible_devices(active_gpus, 'GPU')

    # Initialize the strategy
    strategy = initialize_strategy(config["use_float32"], active_gpus)

    # print tensorflow and keras versions
    logging.info("TensorFlow version: %", tf.__version__)
    logging.info("Keras version: %", tf.keras.__version__)

    return strategy


def setup_logging() -> None:
    """
    Sets up logging configuration for the application.

    Notes
    -----
    This function initializes the Python logging module with a specific format
    and date format. It also removes the default TensorFlow logger handlers to
    prevent duplicate logging and ensures that only the custom configuration is
    used.
    """

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Remove the default Tensorflow logger handlers and use our own
    tf_logger = tf.get_logger()
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()


def initialize_strategy(use_float32: bool,
                        active_gpus: list[str]) -> tf.distribute.Strategy:
    """
    Initializes the TensorFlow distribution strategy and sets the mixed
    precision policy.

    Parameters
    ----------
    use_float32 : bool
        Flag indicating whether to use float32 precision.
    active_gpus : list[str]
        A list of active GPU devices.

    Returns
    -------
    tf.distribute.Strategy
        The initialized TensorFlow distribution strategy.

    Notes
    -----
    This function sets up the MirroredStrategy for distributed training and
    configures the mixed precision policy based on the 'use_float32' and 'gpu'
    arguments. It uses 'mixed_float16' precision when GPUs are used and
    'use_float32' is False, otherwise, it defaults to 'float32' precision.
    """

    # Set the strategy for distributed training
    if len(active_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Detected multiple GPUs, using MirroredStrategy")
    else:
        strategy = tf.distribute.get_strategy()
        logging.info("Using default strategy for single GPU/CPU")

    # Set mixed precision policy
    if not use_float32 and len(active_gpus) > 0:
        # Check if all GPUs support mixed precision
        gpus_support_mixed_precision = bool(active_gpus)
        for device in active_gpus:
            if tf.config.experimental.\
                    get_device_details(device)['compute_capability'][0] < 7:
                gpus_support_mixed_precision = False

        # If all GPUs support mixed precision, enable it
        if gpus_support_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Mixed precision set to 'mixed_float16'")
        else:
            logging.warning(
                "Not all GPUs support efficient mixed precision. Running in "
                "standard mode.")
    else:
        logging.info("Using float32 precision")

    return strategy
