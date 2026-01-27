import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import logging
import unittest
import sys
from pathlib import Path

# > Third party dependencies
import tensorflow as tf
from PIL import Image
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

# Local dependencies
from data.augment_layers import (ShearXLayer,
                                 ElasticTransformLayer,  # noqa: E402
                                 DistortImageLayer, RandomVerticalCropLayer,
                                 RandomWidthLayer, BinarizeLayer,
                                 BlurImageLayer, InvertImageLayer,
                                 InkCorrosionLayer, WaterDamageLayer,
                                 BurnDamageLayer, RestaurationDamageLayer,
                                 MaskingLayer)


class TestDataAugments(unittest.TestCase):
    damage_layers = [InkCorrosionLayer(), WaterDamageLayer(),
                         BurnDamageLayer(), RestaurationDamageLayer()]

    """
    Tests for verifying functionality of image augmentation layers.

    Test coverage:
        1. `test_shear_x_layer`: Tests ShearXLayer for shear transformation
           and shape consistency.
        2. `test_elastic_transform_layer`: Tests ElasticTransformLayer for
           elastic transformations, checking shape consistency.
        3. `test_distort_image_layer`: Tests DistortImageLayer for JPEG-like
           distortions, verifying shape consistency.
        4. `test_random_vertical_crop_layer`: Tests RandomVerticalCropLayer
           for vertical cropping, checking dimension consistency.
        5. `test_random_width_layer`: Tests RandomWidthLayer for width
           adjustment, maintaining dimension consistency.
        6. `test_binarize_otsu`: Tests BinarizeLayer with Otsu method for
           binary thresholding, verifying shape consistency.
        7. `test_binarize_sauvola`: Tests BinarizeLayer with Sauvola method,
           ensuring shape consistency post-binarization.
        8. `test_blur`: Tests BlurImageLayer for applying blur and
           maintaining original shape.
        9. `test_invert`: Tests InvertImageLayer for color inversion and
           shape consistency.
        10.`test_augment_selection`: Tests a pre-set number of augments in
        succession
    """

    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.DEBUG,
        )

        cls.possible_layers = [ShearXLayer(), ElasticTransformLayer(),
                               DistortImageLayer(), RandomVerticalCropLayer(),
                               RandomWidthLayer(), BinarizeLayer('otsu'),
                               BinarizeLayer('sauvola'), BlurImageLayer(),
                               InvertImageLayer()]

    def test_shear_x_layer(self):
        # Test ShearXLayer to ensure it applies a shear transformation
        layer = ShearXLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_elastic_transform_layer(self):
        # Test ElasticTransformLayer for elastic transformation
        layer = ElasticTransformLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_distort_image_layer(self):
        # Test DistortImageLayer for JPEG distortion
        layer = DistortImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_random_vertical_crop_layer(self):
        # Test RandomVerticalCropLayer for random cropping
        layer = RandomVerticalCropLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                # Run in eager mode
                @tf.function(jit_compile=False)
                def call_layer(input_tensor):
                    return layer(input_tensor, training=True)

                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = call_layer(input_tensor)

                # Check that the output shape is different from the input shape
                self.assertNotEqual(input_tensor.shape, output_tensor.shape)

                # Check that only the height has changed
                self.assertEqual(
                    input_tensor.shape[0], output_tensor.shape[0])  # Batch size
                self.assertEqual(
                    input_tensor.shape[2], output_tensor.shape[2])  # Width
                self.assertEqual(
                    input_tensor.shape[3], output_tensor.shape[3])  # Channels

                # Check that the new height is within the expected range
                min_height = int(0.5 * 256)  # Assuming min_factor is 0.5
                max_height = int(1.0 * 256)  # Assuming max_factor is 1.0
                self.assertTrue(
                    min_height <= output_tensor.shape[1] <= max_height)

    def test_random_width_layer(self):
        # Test RandomWidthLayer for width adjustment
        layer = RandomWidthLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                # Run in eager mode
                @tf.function(jit_compile=False)
                def call_layer(input_tensor):
                    return layer(input_tensor, training=True)

                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = call_layer(input_tensor)

                # Check that only the width has changed
                self.assertEqual(
                    input_tensor.shape[0], output_tensor.shape[0])  # Batch size
                self.assertEqual(
                    input_tensor.shape[1], output_tensor.shape[1])  # Height
                self.assertEqual(
                    input_tensor.shape[3], output_tensor.shape[3])  # Channels

                # Check that the new width is within the expected range
                min_width = int(0.5 * 256)  # Assuming min_factor is 0.5
                max_width = int(1.5 * 256)  # Assuming max_factor is 1.5
                self.assertTrue(
                    min_width <= output_tensor.shape[2] <= max_width)

    def test_binarize_otsu(self):
        # Test BinarizeLayer for otsu adjustments and channel check
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = BinarizeLayer(method='otsu', channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_binarize_sauvola(self):
        # Test BinarizeLayer for sauvola adjustments and channel check
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = BinarizeLayer(method='sauvola', channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_blur(self):
        # Test BlurImageLayer for blurring
        layer = BlurImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_invert(self):
        # Test InvertImageLayer for pixel inversion
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = InvertImageLayer(channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_simple_augment_selection(self):
        # Test simple sequential image augments
        augment_selection_a = (self.possible_layers[0:3] +
                               [self.possible_layers[5]])

        # Test without changing image height / width
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                for layer in augment_selection_a:
                    # Check if BinarizeLayer so channels can be set correctly
                    if isinstance(layer, BinarizeLayer):
                        if layer.method == "otsu":
                            layer = BinarizeLayer("otsu",
                                                  channels=channels)
                        else:
                            layer = BinarizeLayer("sauvola",
                                                  channels=channels)

                    output_tensor = layer(input_tensor, training=True)
                    self.assertEqual(input_tensor.shape,
                                     output_tensor.shape)

    def test_hw_changing_augment_selection(self):
        # Test sequential image augments with changing image
        # height and width dimensions
        logging.debug("Testing sequential image augmentations:")
        augment_selection_b = self.possible_layers

        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(
                    shape=[1, 256, 256, channels])
                logging.debug("Initial input size = " +
                              str(input_tensor.shape))
                for layer in augment_selection_b:
                    # Check if BinarizeLayer so channels can be set correctly
                    if isinstance(layer, BinarizeLayer):
                        if layer.method == "otsu":
                            layer = BinarizeLayer("otsu",
                                                  channels=channels)
                        else:
                            layer = BinarizeLayer("sauvola",
                                                  channels=channels)

                    # Apply augment to image in eager mode
                    @tf.function(jit_compile=False)
                    def call_layer(input_tensor):
                        return layer(input_tensor, training=True)

                    output_tensor = call_layer(input_tensor)

                    if layer.name == "random_vertical_crop_layer":
                        # Check that only the height has changed
                        self.assertEqual(
                            input_tensor.shape[0], output_tensor.shape[0])  # Batch size
                        self.assertEqual(
                            input_tensor.shape[2], output_tensor.shape[2])  # Width
                        self.assertEqual(
                            input_tensor.shape[3], output_tensor.shape[3])  # Channels

                        # Check that the new height is within the expected range
                        # Assuming min_factor is 0.5
                        min_height = int(0.5 * input_tensor.shape[1])
                        # Assuming max_factor is 1.0
                        max_height = int(1.0 * input_tensor.shape[1])
                        self.assertTrue(
                            min_height <= output_tensor.shape[1] <= max_height)

                    elif layer.name == "random_width_layer":
                        # Check that only the width has changed
                        self.assertEqual(
                            input_tensor.shape[0], output_tensor.shape[0])  # Batch size
                        self.assertEqual(
                            input_tensor.shape[1], output_tensor.shape[1])  # Height
                        self.assertEqual(
                            input_tensor.shape[3], output_tensor.shape[3])  # Channels

                        # Check that the new width is within the expected range
                        # Assuming min_factor is 0.5
                        min_width = int(0.5 * input_tensor.shape[2])
                        # Assuming max_factor is 1.5
                        max_width = int(1.5 * input_tensor.shape[2])
                        self.assertTrue(
                            min_width <= output_tensor.shape[2] <= max_width)

                    else:
                        self.assertEqual(input_tensor.shape,
                                         output_tensor.shape)

                    logging.debug("Dims after " + layer.name + " = " +
                                  str(output_tensor.shape))
                    input_tensor = output_tensor  # For the next iteration

        logging.debug("Final input dims: " + str(input_tensor.shape))


    def test_damage_layers_shape(self):
        for layer in self.damage_layers:
            for channels in [1, 3, 4]:
                with self.subTest(layer=layer.__class__.__name__, channels=channels):
                    input_tensor = tf.random.uniform(shape=[1, 128, 128, channels])
                    output_tensor = layer(input_tensor, training=True)
                    self.assertEqual(input_tensor.shape, output_tensor.shape,"Problem with layer: " + layer.__class__.__name__)

    def test_damage_layers_no_training(self):
        for layer in self.damage_layers:
            for channels in [1, 3, 4]:
                with self.subTest(layer=layer.__class__.__name__, channels=channels):
                    input_tensor = tf.random.uniform(shape=[128, 128, channels])
                    output_tensor = layer(input_tensor, training=False)
                    self.assertTrue(tf.reduce_all(tf.equal(input_tensor, output_tensor)))

    def test_masking_layer(self):
        # Test MaskingLayer for applying masks and shape consistency.
        layer = MaskingLayer()

        # Load a real image for better visual inspection
        test_image_path = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'test-image1.png'
        image = Image.open(test_image_path).convert('RGB')
        image_np = np.array(image)
        input_tensor = tf.expand_dims(tf.convert_to_tensor(image_np), axis=0)
        input_tensor = tf.cast(input_tensor, tf.float32)

        # Apply the layer
        output_tensor = layer(input_tensor, training=True)

        # Check shape consistency
        self.assertEqual(input_tensor.shape, output_tensor.shape)

        # Save output for visual inspection
        output_image = tf.clip_by_value(output_tensor, 0, 255)
        output_image = tf.cast(output_image, tf.uint8)
        output_image = tf.squeeze(output_image, axis=0)  # Remove batch dimension
        pil_image = Image.fromarray(output_image.numpy())

        output_path = Path('/tmp') / 'test-image1-masked.png'
        pil_image.save(output_path)
        logging.info(f"Saved masked image to {output_path}")

        # Save original for comparison
        original_output_path = Path('/tmp') / 'test-image1-original-for-mask-test.png'
        image.save(original_output_path)
        logging.info(f"Saved original image for mask test to {original_output_path}")


if __name__ == "__main__":
    # load test-image1
    test_image_path = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'test-image1.png'
    # apply all damage layers to the test image and save the results to /tmp/test-image1-damaged-<layer>.png
    test_image = Image.open(test_image_path).convert('RGB')
    # add batch dimension
    test_image = tf.expand_dims(tf.convert_to_tensor(np.array(test_image)), axis=0)

    for layer in TestDataAugments.damage_layers:
        layer_name = layer.__class__.__name__.lower()
        if layer_name == "inkcorrosionlayer":
            layer = InkCorrosionLayer(probability=0.5)
            output_image = layer(tf.convert_to_tensor(np.array(test_image)), training=True)
        else:
            output_image = layer(tf.convert_to_tensor(np.array(test_image)), training=True)
        output_image = tf.clip_by_value(output_image, 0, 255)
        output_image = tf.cast(output_image, tf.uint8)
        output_image = tf.squeeze(output_image, axis=0)  # Remove batch dimension
        output_image = Image.fromarray(output_image.numpy())
        output_path = Path('/tmp') / f'test-image1-damaged-{layer_name}.png'
        output_image.save(output_path)
        logging.info(f"Saved damaged image with {layer_name} to {output_path}")
    # write original image to /tmp/test-image1-original.png
    original_image = Image.fromarray(tf.squeeze(test_image, axis=0).numpy())
    original_output_path = Path('/tmp/test-image1-original.png')
    original_image.save(original_output_path)

    unittest.main()
