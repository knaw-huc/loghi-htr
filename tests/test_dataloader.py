import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import logging
from pathlib import Path
import sys
import unittest

# > Local Dependencies

# > Third party dependencies


class TestDataLoader(unittest.TestCase):
    """
    Tests for the data_generator class.

    Test coverage:
        1. `test_initialization` tests that the instance variables are
        initialized correctly.
        2. `test_load_images` test shapes before and after pre-processing
        and encoding of the label.
        3. `test_load_images_with_augmentation` tests the shapes before and
        after pre-processing and encoding of the label when augmentation is
        applied.
    """

    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        # Add the src directory to the path
        sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

        from data.loader import DataLoader
        cls.DataLoader = DataLoader

        from utils.text import Tokenizer
        cls.Tokenizer = Tokenizer

        from data.augmentation import ResizeWithPadLayer
        cls.ResizeWithPadLayer = ResizeWithPadLayer

    def test_initialization(self):
        tokenizer = self.Tokenizer(tokens=list("ABC"))
        dg = self.DataLoader(tokenizer=tokenizer, height=64, channels=1)

        # Verify that the instance variables are initialized correctly.
        self.assertEqual(dg.height, 64)
        self.assertEqual(dg.channels, 1)

    def test_load_images(self):
        images = [
            "tests/data/test-image1.png",
            "tests/data/test-image2.png",
            "tests/data/test-image3.png",
        ]
        sample_weights = ["1.0", "0.0", "0.5"]
        labels = []
        for image in images:
            image_label_loc = image.replace("png", "txt")
            with open(image_label_loc, "r") as f:
                labels.append(f.read())

        vocab = list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,")

        image_info_tuples = list(zip(images, labels, sample_weights))

        tokenizer = self.Tokenizer(tokens=vocab)
        dg = self.DataLoader(tokenizer=tokenizer, height=64, channels=1)

        for image_info_tuple in image_info_tuples:
            # Mock TensorFlow's file reading and decoding operations
            preprocessed_image, encoded_label, sample_weight \
                = dg.process_sample(image_info_tuple)

            # Assert the shape of the preprocessed image
            self.assertEqual(preprocessed_image.shape[1], 64)
            self.assertEqual(preprocessed_image.shape[2], 1)

            # Assert correct encoding of the label
            decoded_label = tokenizer.decode(encoded_label)
            self.assertEqual(decoded_label, image_info_tuple[1])

            # Assert the sample weights
            self.assertEqual(sample_weight, float(image_info_tuple[2]))

if __name__ == '__main__':
    unittest.main()
