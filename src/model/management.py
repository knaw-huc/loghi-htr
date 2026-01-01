# Imports

# > Standard library
import logging
import os
from typing import Any, Dict, Optional
import json
import shutil
import warnings
import zipfile

# > Third-party dependencies
import tensorflow as tf
from vgslify.generator import VGSLModelGenerator

# > Local dependencies
from model.custom_model import build_custom_model
from model.conversion import convert_model
from model.replacing import replace_final_layer, replace_recurrent_layer
from setup.config import Config
from utils.text import Tokenizer
from datetime import datetime


def adjust_model_for_float32(model: tf.keras.Model) -> tf.keras.Model:
    """
    Adjusts a given Keras model to use float32 data type for all layers.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be adjusted for float32.

    Returns
    -------
    tf.keras.Model
        A new model with all layers set to use float32 data type.
    """

    # Recreate the exact same model but with float32
    config = model.get_config()

    # Set the dtype policy for each layer in the configuration
    for layer_config in config['layers']:
        if 'dtype' in layer_config['config']:
            layer_config['config']['dtype'] = 'float32'
        if 'dtype_policy' in layer_config['config']:
            layer_config['config']['dtype_policy'] = {
                'class_name': 'Policy',
                'config': {'name': 'float32'}}

    # Create a new model from the modified configuration
    model_new = tf.keras.Model.from_config(config)
    model_new.set_weights(model.get_weights())
    model = model_new

    # Verify float32
    for layer in model.layers:
        if layer.dtype != 'float32':
            logging.error("Layer %s is not float32", layer.name)

    return model


def customize_model(model: tf.keras.Model,
                    config: Config,
                    tokenizer: Tokenizer) -> tf.keras.Model:
    """
    Customizes a Keras model based on various arguments including layer
    replacement and freezing options.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be customized.
    config : Config
        A set of arguments controlling how the model should be customized.
    tokenizer : Tokenizer
        The tokenizer object used for tokenization.

    Returns
    -------
    tf.keras.Model
        The customized model.
    """

    # Replace certain layers if specified
    if config["replace_recurrent_layer"]:
        logging.info("Replacing recurrent layer with %s",
                     config["replace_recurrent_layer"])
        model = replace_recurrent_layer(model,
                                        len(tokenizer),
                                        config["replace_recurrent_layer"])

    # Replace the final layer if specified
    if config["replace_final_layer"] or not os.path.isdir(config["model"]):
        new_classes = len(tokenizer)
        logging.info("Replacing final layer with %s classes", new_classes)
        model = replace_final_layer(model, len(tokenizer), model.name)

    # Freeze layers if specified
    if any([config["freeze_conv_layers"],
            config["freeze_recurrent_layers"],
            config["freeze_dense_layers"]]):
        for layer in model.layers:
            if config["freeze_conv_layers"] and \
                (layer.name.lower().startswith("conv") or
                 layer.name.lower().startswith("residual")):
                logging.info("Freezing layer: %s", layer.name)
                layer.trainable = False
            elif config["freeze_recurrent_layers"] and \
                    layer.name.lower().startswith("bidirectional"):
                logging.info("Freezing layer: %s", layer.name)
                layer.trainable = False
            elif config["freeze_dense_layers"] and \
                    layer.name.lower().startswith("dense"):
                logging.info("Freezing layer: %s", layer.name)
                layer.trainable = False

    # Further configuration based on use_float32 and gpu
    if config["use_float32"] or config["gpu"] == "-1":
        # Adjust the model for float32
        logging.info("Adjusting model for float32")
        model = adjust_model_for_float32(model)

    return model


def load_model_from_directory(directory: str,
                              output_directory: Optional[str] = None,
                              custom_objects: Optional[Dict[str, Any]] = None,
                              compile: bool = True) -> tf.keras.Model:
    """
    Load a TensorFlow Keras model from a specified directory.

    This function supports loading models in both the SavedModel format (.pb)
    and the Keras format (.keras). It first searches for a .pb file to identify
    a SavedModel. If not found, it looks for a .keras file.

    Parameters
    ----------
    directory : str
        The directory where the model is saved.
    output_directory : Optional[str], optional
        The directory where the model should be saved after conversion, by
        default None.
    custom_objects : Optional[Dict[str, Any]], optional
        Optional dictionary mapping names (strings) to custom classes or
        functions to be considered during deserialization, by default None.
    compile : bool, optional
        Whether to compile the model after loading, by default True.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If no suitable model file is found in the specified directory.
    """

    # Check for a .pb file (indicating SavedModel format)
    if any(file.endswith('.pb') for file in os.listdir(directory)):
        return convert_model(directory, output_directory, custom_objects)

    # Look for a .keras file
    model_file = next((os.path.join(directory, file) for file in os.listdir(
        directory) if file.endswith(".keras")), None)

    if model_file:
        try:
            return tf.keras.models.load_model(model_file,
                                              custom_objects=custom_objects,
                                              compile=compile)
        except (TypeError, ValueError) as err:
            logging.error(err)
            logging.error("Error loading model. Attempting to convert the model to the new format.")

            # Convert the old model to the new format
            model = _convert_old_model_to_new(model_file, custom_objects,
                                              compile=compile)

            # Save the converted model
            # Rename the old model file
            if not os.path.exists(model_file + ".old"):
                logging.info("Renaming old model file to %s",
                             model_file + ".old")
                old_model_file = model_file + ".old"
                os.rename(model_file, old_model_file)

            # Save the new model
            logging.info("Saving new model to %s", model_file)
            model.save(model_file)

            return model

    raise FileNotFoundError("No suitable model file found in the directory.")


def _convert_old_model_to_new(model_file: str,
                              custom_objects: dict,
                              compile: bool = True) -> tf.keras.Model:
    """
    Converts an old v2 Keras model to the new v3 format.

    Parameters
    ----------
    model_file : str
        The path to the .keras file containing the old model.
    custom_objects : dict
        Custom objects required for model loading.
    compile : bool, optional
        Whether to compile the model after loading, by default True.

    Returns
    -------
    tf.keras.Model
        The converted Keras model.
    """

    # Temporary directory to extract the .keras file contents
    temp_dir = "/tmp/keras_model_extraction"

    # Ensure the temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Extract .keras file
    with zipfile.ZipFile(model_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Load model architecture from config.json
    with open(os.path.join(temp_dir, 'config.json'), 'r') as json_file:
        model_config = json.load(json_file)

    model_config["module"] = "keras.src.models.functional"

    # Function to recursively correct "axis" in BatchNormalization layers
    def correct_axis(layer_config):
        if isinstance(layer_config, dict):
            if layer_config.get('class_name') == 'BatchNormalization' \
                    and isinstance(layer_config.get('config', {})
                                   .get('axis'), list):
                layer_config['config']['axis'] \
                    = layer_config['config']['axis'][0]
            elif layer_config.get('class_name') == 'Bidirectional':
                lstm_layer_config = layer_config['config']['layer']['config']
                lstm_layer_config['recurrent_initializer']['class_name'] \
                    = 'OrthogonalInitializer'
                lstm_layer_config.pop('time_major', None)
            elif 'layers' in layer_config:
                for sub_layer in layer_config['layers']:
                    correct_axis(sub_layer)

    correct_axis(model_config["config"])

    # Replace 'Policy' with 'DTypePolicy' in all layers' dtype configurations
    def replace_policy_with_dtypepolicy(obj):
        if isinstance(obj, dict):
            if obj.get('class_name') == 'Policy':
                obj['class_name'] = 'DTypePolicy'
            for key, value in obj.items():
                replace_policy_with_dtypepolicy(value)
        elif isinstance(obj, list):
            for item in obj:
                replace_policy_with_dtypepolicy(item)

    replace_policy_with_dtypepolicy(model_config["config"]['layers'])

    if model_config.get("compile_config"):
        compile_optimizer = model_config["compile_config"]["optimizer"]
        compile_optimizer["module"] = "keras.optimizers"
        compile_optimizer["config"].pop("jit_compile", None)
        compile_optimizer["config"].pop("is_legacy_optimizer", None)

    if not compile:
        model_config.pop("compile_config", None)

    # Save the corrected config back to a temporary json file
    corrected_config_path = os.path.join(temp_dir, 'corrected_config.json')
    with open(corrected_config_path, 'w') as json_file:
        json.dump(model_config, json_file)

    # Load model from corrected config
    with open(corrected_config_path, 'r') as json_file:
        corrected_model_json = json_file.read()
    model = tf.keras.models.model_from_json(
        corrected_model_json, custom_objects=custom_objects)

    # Load weights into the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_weights(os.path.join(temp_dir, 'model.weights.h5'),
                           skip_mismatch=True)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    return model


def load_or_create_model(config: Config, custom_objects: Dict[str, Any]) -> tf.keras.Model:
    """
    Loads an existing Keras model or creates a new one based on the provided configuration.

    Parameters
    ----------
    config : Config
        Configuration object containing model and other arguments.
    custom_objects : Dict[str, Any]
        Dictionary of custom objects required for model loading.

    Returns
    -------
    tf.keras.Model
        The loaded or newly created Keras model.
    """

    model_path = config["model"]

    # Check if the provided model is a directory (indicating a saved model)
    if os.path.isdir(model_path):
        model = load_model_from_directory(
            model_path, output_directory=config["output"], custom_objects=custom_objects)
        if config["model_name"]:
            model._name = config["model_name"]

    # Handle 'custom' model type
    elif model_path == 'custom':
        model = build_custom_model()
    else:
        model = build_predefined_model(config, model_path)

    return model


def build_predefined_model(config: Config, model_key: str) -> tf.keras.Model:
    """
    Builds a model from the predefined library based on the configuration.

    Parameters
    ----------
    config : Config
        Configuration object containing model specifications.
    model_key : str
        Key for selecting the model from the predefined library.

    Returns
    -------
    tf.keras.Model
        The newly built Keras model.
    """

    model_library = get_model_library()
    model_spec = model_library.get(model_key, model_key)

    if model_spec == model_key:
        if not config["model_name"]:
            # append date and time to model name if not specified
            config["model_name"] = ('vgsl_model_' + datetime.now().strftime("%Y%m%d_%H%M%S"))

    model_generator = VGSLModelGenerator()
    model = model_generator.generate_model(model_spec=model_spec,
                                           model_name=config["model_name"] if config["model_name"]
                                           else model_key)

    return model


def get_model_library() -> dict:
    """
    Returns a dictionary of predefined models with their VGSL spec strings.

    Returns
    -------
    dict
        Dictionary of predefined models with their VGSL spec strings.
    """

    model_library = {
        "modelkeras":
            ("None,None,64,1 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc3 "
             "Fl64 D20 Bl128 D20 Bl64 D20 Fs92"),
        "model9":
            ("None,None,64,1 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 "
             "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc3 Bl256,D50 Bl256,D50 "
             "Bl256,D50 Bl256,D50 Bl256,D50 Fs92"),
        "model10":
            ("None,None,64,4 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 "
             "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc3 Bl256,D50 Bl256,D50 "
             "Bl256,D50 Bl256,D50 Bl256,D50 Fs92"),
        "model11":
            ("None,None,64,1 Cr3,3,24 Bn Ap2,2,2,2 Cr3,3,48 Bn Cr3,3,96 Bn"
             "Ap2,2,2,2 Cr3,3,96 Bn Ap2,2,2,2 Rc3 Bl256 Bl256 Bl256 "
             "Bl256 Bl256 Fe1024 Fs92"),
        "model12":
            ("None,None,64,1 Cr1,3,12 Bn Cr3,3,48 Bn Mp2,2,2,2 Cr3,3,96 "
             "Cr3,3,96 Bn Mp2,2,2,2 Rc3 Bl256 Bl256 Bl256 Bl256 Bl256 "
             "Fs92"),
        "model13":
            ("None,None,64,1 Cr1,3,12 Bn Cr3,1,24 Bn Mp2,2,2,2 Cr1,3,36 "
             "Bn Cr3,1,48 Bn Cr1,3,64 Bn Cr3,1,96 Bn Cr1,3,96 Bn Cr3,1,96 "
             "Bn Rc3 Bl256 Bl256 Bl256 Bl256 Bl256 Fs92"),
        "model14":
            ("None,None,64,1 Ce3,3,24 Bn Mp2,2,2,2 Ce3,3,36 Bn Mp2,2,2,2 "
             "Ce3,3,64 Bn Mp2,2,2,2 Ce3,3,96 Bn Ce3,3,128 Bn Rc3 Bl256,D50 "
             "Bl256,D50 Bl256,D50 Bl256,D50 Bl256,D50 Fs92"),
        "model15":
            ("None,None,64,1 Ce3,3,8 Bn Mp2,2,2,2 Ce3,3,12 Bn Ce3,3,20 Bn "
             "Ce3,3,32 Bn Ce3,3,48 Bn Rc3 Bg256,D50 Bg256,D50 Bg256,D50 "
             "Bg256,D50 Bg256,D50 Fs92"),
        "model16":
            ("None,None,64,1 Ce3,3,8 Bn Mp2,2,2,2 Ce3,3,12 Bn Ce3,3,20 Bn "
             "Ce3,3,32 Bn Ce3,3,48 Bn Rc3 Lfs128,D50 Lfs128,D50 Lfs128,D50 "
             "Lfs128,D50 Lfs128,D50 Fs92"),
        "recommended":
            ("None,None,64,1 Cr3,3,24 Mp2,2,2,2 Bn Cr3,3,48 Bn Cr3,3,96 "
             "Mp2,2,2,2 Bn Cr3,3,96 Mp2,2,2,2 Bn Rc3 Bl512 D50 Bl512 D50 "
             "Bl512 D50 Bl512 D50 Bl512 D50 Fs92")
    }

    return model_library
