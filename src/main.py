# Imports
import argparse
# > Standard library
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import time
import logging

# > Local dependencies
# Data handling
from data.data_handling import initialize_data_manager

# Model-specific
from data.augmentation import make_augment_model, visualize_augments
from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.metrics import CERMetric, WERMetric
from model.management import load_or_create_model, customize_model
from model.optimization import create_learning_rate_schedule, get_optimizer, \
    LoghiLearningRateSchedule
from modes.training import train_model, plot_training_history
from modes.evaluation import perform_evaluation

# Setup and configuration
from setup.arg_parser import get_args
from setup.config import Config
from setup.environment import setup_environment, setup_logging

# Utilities
from utils.print import summarize_model
from utils.text import Tokenizer


def main(args=None):
    """ Main function for the program """
    setup_logging()

    # Get the arguments
    if args is None:
        parsed_args = get_args()
        print('parsed_args: ', parsed_args)
    else:
        parsed_args = get_args(args)
        # parsed_args = args
    config = Config(*parsed_args)

    # Set up the environment
    strategy = setup_environment(config)

    # Create the output directory if it doesn't exist
    if config["output"]:
        os.makedirs(config["output"], exist_ok=True)

    # Determine the path to the tokenizer file
    json_path = None

    if config["tokenizer"]:
        json_path = config["tokenizer"]
    elif os.path.isdir(config["model"]):
        json_path = next(
            (os.path.join(config["model"], fname) for fname in ["tokenizer.json", "charlist.txt"]
             if os.path.exists(os.path.join(config["model"], fname))),
            None
        )

    # Load the tokenizer if a valid path was found
    if json_path and not config["replace_final_layer"]:
        tokenizer = Tokenizer.load_from_file(json_path)
    else:
        tokenizer = None  # Indicate that a new tokenizer will be created later

    # Set the custom objects
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(config, custom_objects)
        augmentation_model = make_augment_model(config, model.input_shape[-1])

        if config["visualize_augments"] and augmentation_model:
            visualize_augments(augmentation_model,
                               config["output"],
                               model.input_shape[-1])

        # Initialize the DataManager
        data_manager = initialize_data_manager(config, tokenizer, model,
                                               augmentation_model)

        # Replace the tokenizer with the one from the data manager
        tokenizer = data_manager.tokenizer
        logging.info("Tokenizer size: %s tokens", len(tokenizer))

        # Additional model customization such as freezing layers, replacing
        # layers, or adjusting for float32
        model = customize_model(model, config, tokenizer)

        # Save the tokenizer
        tokenizer.save_to_json(os.path.join(config["output"],
                                            "tokenizer.json"))

        # Create the learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            learning_rate=config["learning_rate"],
            decay_rate=config["decay_rate"],
            decay_steps=config["decay_steps"],
            train_batches=data_manager.get_train_batches(),
            do_train=config["train_list"],
            warmup_ratio=config["warmup_ratio"],
            epochs=config["epochs"],
            decay_per_epoch=config["decay_per_epoch"],
            linear_decay=config["linear_decay"])

        # Create the optimizer
        optimizer = get_optimizer(config["optimizer"], lr_schedule)

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss=CTCLoss(),
                      metrics=[CERMetric(greedy=config["greedy"],
                                         beam_width=config["beam_width"]),
                               WERMetric()],
                      weighted_metrics=[])

    # Print the model summary
    logging.info("Model Summary:")
    model.summary(line_length=100)

    # Store the model info (i.e., git hash, args, model summary, etc.)
    config.update_config_key("model", summarize_model(model))
    config.update_config_key("model_name", model.name)
    config.update_config_key("model_channels", model.input_shape[-1])
    config.save()

    # Store timestamps
    timestamps = {'start_time': time.time()}

    # Train the model
    if config["train_list"]:
        tick = time.time()

        history = train_model(model,
                              config,
                              data_manager.datasets["train"],
                              data_manager.datasets["evaluation"],
                              data_manager)
        # Plot the training history
        plot_training_history(history=history,
                              output_path=config["output"],
                              plot_validation=bool(config["validation_list"]))

        timestamps['Training'] = time.time() - tick

    # Evaluation modes and their corresponding conditions
    evaluation_modes = [
        ("validation", config["do_validate"],
         "Validation results are without special markdown tags"),
        ("test", config["test_list"],
         "Test results are without special markdown tags"),
        ("inference", config["inference_list"], None)
    ]

    for mode, condition, warning in evaluation_modes:
        if condition:
            if warning:
                logging.warning(warning)
            tick = time.time()
            perform_evaluation(config, model, data_manager, mode)
            timestamps[mode.capitalize()] = time.time() - tick

    # Log the timestamps
    logging.info("--------------------------------------------------------")
    for key, value in list(timestamps.items())[1:]:
        logging.info("%s completed in %.2f seconds", key, value)
    logging.info("Total time: %.2f seconds",
                 time.time() - timestamps['start_time'])


if __name__ == "__main__":
    main()
