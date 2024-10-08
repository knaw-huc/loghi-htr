# Imports

# > Standard library
import os
import time
import logging

# > Local dependencies
# Data handling
from data.data_handling import save_charlist, load_initial_charlist, \
    initialize_data_manager

# Model-specific
from data.augmentation import make_augment_model, visualize_augments
from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.metrics import CERMetric, WERMetric
from model.management import load_or_create_model, customize_model, \
    verify_charlist_length
from model.optimization import create_learning_rate_schedule, get_optimizer, \
    LoghiLearningRateSchedule
from modes.training import train_model, plot_training_history
from modes.validation import perform_validation
from modes.test import perform_test
from modes.inference import perform_inference

# Setup and configuration
from setup.arg_parser import get_args
from setup.config import Config
from setup.environment import setup_environment, setup_logging

# Utilities
from utils.print import summarize_model


def main():
    """ Main function for the program """
    setup_logging()

    # Get the arguments
    parsed_args = get_args()
    config = Config(*parsed_args)

    # Set up the environment
    strategy = setup_environment(config)

    # Create the output directory if it doesn't exist
    if config["output"]:
        os.makedirs(config["output"], exist_ok=True)

    # Get the initial character list
    if os.path.isdir(config["model"]) or config["charlist"]:
        charlist, removed_padding = load_initial_charlist(
            config["charlist"], config["model"],
            config["output"], config["replace_final_layer"])
    else:
        charlist = []
        removed_padding = False

    # Set the custom objects
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(config, custom_objects)
        augmentation_model = make_augment_model(config, model.input_shape[-1])

        if config["visualize_augments"]:
            visualize_augments(augmentation_model, config["output"],
                               model.input_shape[-1])

        # Initialize the DataManager
        data_manager = initialize_data_manager(config, charlist, model,
                                               augmentation_model)

        # Replace the charlist with the one from the data manager
        charlist = data_manager.charlist
        logging.info("Using charlist: %s", charlist)
        logging.info("Charlist length: %s", len(charlist))

        # Additional model customization such as freezing layers, replacing
        # layers, or adjusting for float32
        model = customize_model(model, config, charlist)

        # Save the charlist
        verify_charlist_length(charlist, model, config["use_mask"],
                               removed_padding)
        save_charlist(charlist, config["output"])

        # Create the learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            learning_rate=config["learning_rate"],
            decay_rate=config["decay_rate"],
            decay_steps=config["decay_steps"],
            train_batches=data_manager.get_train_batches(),
            do_train=config["do_train"],
            warmup_ratio=config["warmup_ratio"],
            epochs=config["epochs"],
            decay_per_epoch=config["decay_per_epoch"],
            linear_decay=config["linear_decay"])

        # Create the optimizer
        optimizer = get_optimizer(config["optimizer"], lr_schedule)

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss=CTCLoss,
                      metrics=[CERMetric(greedy=config["greedy"],
                                         beam_width=config["beam_width"]),
                               WERMetric()],
                      weighted_metrics=[])

    # Print the model summary
    model.summary()

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
        plot_training_history(history=history, output_path=config["output"],
                              plot_validation=bool(config["validation_list"]))

        timestamps['Training'] = time.time() - tick

    # Evaluate the model
    if config["do_validate"]:
        logging.warning("Validation results are without special markdown tags")

        tick = time.time()
        perform_validation(config, model, charlist, data_manager)
        timestamps['Validation'] = time.time() - tick

    # Test the model
    if config["test_list"]:
        logging.warning("Test results are without special markdown tags")

        tick = time.time()
        perform_test(config, model, data_manager.datasets["test"],
                     charlist, data_manager)
        timestamps['Test'] = time.time() - tick

    # Infer with the model
    if config["inference_list"]:
        tick = time.time()
        perform_inference(config, model, data_manager.datasets["inference"],
                          charlist, data_manager)
        timestamps['Inference'] = time.time() - tick

    # Log the timestamps
    logging.info("--------------------------------------------------------")
    for key, value in list(timestamps.items())[1:]:
        logging.info("%s completed in %.2f seconds", key, value)
    logging.info("Total time: %.2f seconds",
                 time.time() - timestamps['start_time'])


if __name__ == "__main__":
    main()
