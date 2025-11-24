# Imports

# > Standard library
import logging

# > Third party dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from vgslify.generator import VGSLModelGenerator


def replace_recurrent_layer(model: tf.keras.Model,
                            number_characters: int,
                            vgsl_string: str) -> tf.keras.Model:
    """
    Replace recurrent layers in a given Keras model with new layers specified
    by a VGSL string.

    Parameters
    ----------
    model : tf.keras.Model
        The original Keras model in which recurrent layers need to be replaced.
    number_characters : int
        The number of characters/classes for the Dense layer before the
        activation.
    vgsl_string : str
        The VGSL spec string that defines the new layers to replace the
        recurrent ones.

    Returns
    -------
    tf.keras.Model
        A new Keras model with the recurrent layers replaced by those specified
        in the VGSL string.

    Raises
    ------
    ValueError:
        If no recurrent layers are found in the given model.
    """

    logging.info("Starting the replacement of recurrent layers in the model.")

    initializer = tf.keras.initializers.GlorotNormal()

    # Identify layers up to the first recurrent layer
    last_layer = None
    found_rnn = False
    for layer in model.layers:
        if isinstance(layer, (layers.GRU, layers.LSTM, layers.Bidirectional)):
            found_rnn = True
            break
        last_layer = layer

    if not found_rnn:
        logging.error("No recurrent layers found in the model.")
        raise ValueError("No recurrent layers found in the model.")

    # Generate new layers using VGSLModelGenerator
    logging.info("Generating new layers using VGSLModelGenerator.")
    vgsl_string = "None,None " + vgsl_string
    history = VGSLModelGenerator().generate_history(vgsl_string)

    logging.debug("VGSLModelGenerator history: %s", history)

    # Add the new layers to the model
    x = last_layer.output
    for layer in history[1:]:
        x = layer(x)

    dense_layer_name = model.layers[-2].name
    x = layers.Dense(number_characters,
                     activation="softmax",
                     name=dense_layer_name,
                     kernel_initializer=initializer)(x)
    output = layers.Activation('linear', dtype=tf.float32)(x)

    old_model_name = model.name
    model = keras.models.Model(
        inputs=model.input, outputs=output, name=old_model_name
    )

    logging.info("Recurrent layers replaced successfully in model: %s.",
                 old_model_name)

    return model


def replace_final_layer(model: tf.keras.models.Model,
                        number_characters: int,
                        model_name: str) -> tf.keras.models.Model:
    """
    Replace the final layer of a given Keras model.

    This function replaces the final dense layer of a model with a new dense
    layer that has a specified number of units. An optional mask can be used
    which affects the number of units in the new dense layer.

    Parameters
    ----------
    model : tf.keras.models.Model
        The Keras model whose final layer is to be replaced.
    number_characters : int
        Number of units for the new dense layer.
    model_name : str
        Name to assign to the modified model.

    Returns
    -------
    tf.keras.models.Model
        The modified Keras model with a new final layer.
    """

    # Initialize the kernel weights
    initializer = tf.keras.initializers.GlorotNormal()

    # Find the name of the last layer before the dense or activation layers
    last_layer = ""
    activation_layers = []
    for layer in model.layers:
        if not layer.name.startswith(("dense", "activation")):
            last_layer = layer.name
        elif layer.name.startswith("activation"):
            activation_layers.append(layer.name)

    x = layers.Dense(number_characters, name="dense_out",
                     kernel_initializer=initializer)(model.get_layer(last_layer).output)

    # Add a softmax activation layer with float32 data type
    new_name = "activation"
    base_name= new_name
    i = 0
    while new_name in activation_layers:
        i += 1
        new_name = base_name +"_" + str(i)

    output = layers.Activation('softmax', dtype=tf.float32, name=new_name)(x)

    # Construct the final model
    new_model = tf.keras.models.Model(
        inputs=model.input, outputs=output, name=model_name
    )

    return new_model
