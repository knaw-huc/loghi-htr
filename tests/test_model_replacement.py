import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
from pathlib import Path
import sys
import logging
import unittest

# > Third party dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class TestReplaceLayers(unittest.TestCase):
    """
    Tests for the `replace_recurrent_layer` function in the `model` module.

    The tests are designed to check that the function correctly replaces RNN
    layers in a model with the layers specified in the VGSL string. The
    function should also correctly handle dropout layers specified in the VGSL
    string.

    Test coverage:
        1. `test_rnn_replacement_simple`: Test that the function correctly
            replaces a simple LSTM layer with the layers specified in the VGSL
            string.
        2. `test_rnn_replacement_gru`: Test that the function correctly
            replaces a GRU layer with the layers specified in the VGSL string.
        3. `test_rnn_replacement_lstm`: Test that the function correctly
            replaces a LSTM layer with the layers specified in the VGSL string.
        4. `test_rnn_replacement_bidirectional`: Test that the function
            correctly replaces a Bidirectional LSTM layer with the layers
            specified in the VGSL string.
        5. `test_rnn_replacement_multiple`: Test that the function correctly
            replaces multiple RNN layers with the layers specified in the VGSL
            string.
        6. `test_rnn_replacement_with_dropout`: Test that the function
            correctly replaces RNN layers with the layers specified in the VGSL
            string, and also correctly handles dropout layers.
        7. `test_no_rnn_error_handling`: Test that the function correctly
            raises a ValueError when no RNN layers are found in the model.
    """

    @classmethod
    def setUpClass(cls):
        # Add the path to the project to the system path
        sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        import model.replacing as replacing
        cls.replacing = replacing

    def test_rnn_replacement_simple(self):
        # Create a simple model to test
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, (3, 3))(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Reshape((-1, 1024))(x)
        x = layers.Dense(64)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Use the function to replace layers
        new_model = self.replacing.replace_recurrent_layer(
            test_model, 29, "Lfs128 Lf64")

        # Check that the model is not None
        self.assertIsNotNone(new_model, "Model is None after replacement")

    def test_rnn_replacement_gru(self):
        # Create a simple model to test
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Reshape((-1, 49*31*32))(x)
        x = layers.GRU(128, return_sequences=True)(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Use the function to replace GRU layer
        new_model = self.replacing.replace_recurrent_layer(
            test_model, 29, "Lfs128")

        # Check the last two layers of the new model
        self.assertIsInstance(
            new_model.layers[-2], layers.Dense, "Second to last layer is not "
            "a Dense layer")
        self.assertIsInstance(
            new_model.layers[-1], layers.Activation, "Last layer is not an "
            "Activation layer")

        # Check that the replaced layers are as expected
        found_lstm = False
        for layer in new_model.layers:
            if 'gru' in layer.name:
                self.assertNotIsInstance(
                    layer, layers.GRU, "GRU layer was not replaced")
            if 'lstm' in layer.name and isinstance(layer, layers.LSTM):
                found_lstm = True

        self.assertTrue(found_lstm, "GRU layer was not replaced")

        # Forward Pass
        try:
            dummy_input = np.random.rand(1, 100, 64, 3)
            _ = new_model.predict(dummy_input)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {str(e)}")

    def test_rnn_replacement_lstm(self):
        # Create a simple model to test
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Reshape((-1, 49*31*32))(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Use the function to replace LSTM layer
        new_model = self.replacing.replace_recurrent_layer(
            test_model, 29, "Gfs128")

        # Check the last two layers of the new model
        self.assertIsInstance(
            new_model.layers[-2], layers.Dense, "Second to last layer is not "
            "a Dense layer")
        self.assertIsInstance(
            new_model.layers[-1], layers.Activation, "Last layer is not an "
            "Activation layer")

        # Check that the replaced layers are as expected
        found_gru = False
        for layer in new_model.layers:
            self.assertNotIsInstance(
                layer, layers.LSTM, "LSTM layer was not replaced")
            if 'gru' in layer.name and isinstance(layer, layers.GRU):
                found_gru = True

        self.assertTrue(found_gru, "LSTM layer was not replaced")

        # Forward Pass
        try:
            dummy_input = np.random.rand(1, 100, 64, 3)
            _ = new_model.predict(dummy_input)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {str(e)}")

    def test_rnn_replacement_bidirectional(self):
        # Create a simple model to test
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Reshape((-1, 49*31*32))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Use the function to replace Bidirectional layer
        new_model = self.replacing.replace_recurrent_layer(
            test_model, 29, "Lfs128 Lf64")

        # Check the last two layers of the new model
        self.assertIsInstance(
            new_model.layers[-2], layers.Dense, "Second to last layer is not a"
            " Dense layer")
        self.assertIsInstance(
            new_model.layers[-1], layers.Activation, "Last layer is not an "
            "Activation layer")

        # Check that the replaced layers are as expected
        for layer in new_model.layers:
            if 'bidirectional' in layer.name:
                self.fail(
                    "Old Bidirectional layers should not be present in the new"
                    " model.")

        # Forward Pass
        try:
            dummy_input = np.random.rand(1, 100, 64, 3)
            _ = new_model.predict(dummy_input)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {str(e)}")

    def test_rnn_replacement_multiple(self):
        # Create a simple model with LSTM, GRU, and Bidirectional LSTM layers
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Reshape((-1, 32 * 32))(x)  # Flatten the spatial dimensions
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Use the function to replace RNN layers
        vgsl_string = "Lfs128 Gfs64 Bl32"
        new_model = self.replacing.replace_recurrent_layer(
            test_model, 29, vgsl_string)

        # Assertions to check layer types and configurations
        found_lstm, found_gru, found_bidir = False, False, False
        for layer in new_model.layers:
            if isinstance(layer, layers.LSTM):
                self.assertEqual(layer.units, 128,
                                 "Unexpected number of units in LSTM layer")
                found_lstm = True
            elif isinstance(layer, layers.GRU):
                self.assertEqual(
                    layer.units, 64, "Unexpected number of units in GRU layer")
                found_gru = True
            elif isinstance(layer, layers.Bidirectional):
                self.assertEqual(
                    layer.forward_layer.units, 32, "Unexpected number of units in "
                    "Bidirectional LSTM layer")
                found_bidir = True

        self.assertTrue(found_lstm, "LSTM layer not found in the new model")
        self.assertTrue(found_gru, "GRU layer not found in the new model")
        self.assertTrue(
            found_bidir, "Bidirectional LSTM layer not found in the new model")

    def test_rnn_replacement_with_dropout(self):
        # Create a simple model with LSTM and GRU layers
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Reshape((-1, 32 * 32))(x)  # Flatten the spatial dimensions
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Use the function to replace RNN layers and include dropout
        vgsl_string = "Lfs128 D10 Gf64 D15"
        new_model = self.replacing.replace_recurrent_layer(
            test_model, 29, vgsl_string)

        # Assertions to check dropout layers
        dropout_rates = [0.1, 0.15]
        dropout_index = 0
        for layer in new_model.layers:
            if isinstance(layer, layers.Dropout):
                self.assertEqual(
                    layer.rate, dropout_rates[dropout_index], "Unexpected "
                    f"dropout rate for Dropout layer {layer.name}. Expected "
                    f"{dropout_rates[dropout_index]}, got {layer.rate}")
                dropout_index += 1

        self.assertEqual(dropout_index, len(dropout_rates),
                         "Mismatch in number of Dropout layers")

    def test_no_rnn_error_handling(self):
        # Create a simple model without any RNN layers
        inputs = layers.Input(shape=(None, 64, 3), name="image")
        x = layers.Conv2D(32, (3, 3))(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, (3, 3))(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dense(29, activation="softmax")(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        test_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Assert that the ValueError is raised with the expected error message
        with self.assertRaises(ValueError) as context:
            self.replacing.replace_recurrent_layer(test_model, 29, "Lfs128")

        self.assertEqual(str(context.exception),
                         "No recurrent layers found in the model.")


if __name__ == "__main__":
    unittest.main()
