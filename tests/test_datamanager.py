import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import logging
from pathlib import Path
import sys
import tempfile
import unittest

# > Third party dependencies
import tensorflow as tf


class DataManagerTest(unittest.TestCase):
    """
    Tests for the DataManager class.

    Test Coverage:
        1. `test_initialization`: Checks the correct instantiation of the
        DataManager class and its default values.
        2. `test_create_data_simple`: Checks if the create_data function
        works as expected.
        3. `test_missing_files`: Tests the behavior when data lists contain
        references to non-existent files.
        4. `test_unsupported_chars`: Validates the handling of labels with
        unsupported characters.
        5. `test_inference_mode`: Checks the DataManager"s behavior in
        inference mode.
        6. `test_text_normalization`: Verifies the correct normalization of
        text labels.
        7. `test_multiplication`: Tests the multiplication functionality for
        increasing dataset size.
        8. `test_generators`: Validates the creation and behavior of data
        generators for training, validation, test, and inference.
    """

    @classmethod
    def setUpClass(cls):

        # Reset the default graph before each test
        tf.compat.v1.reset_default_graph()

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        cls.dummy_config = {
            "do_validate": False,
            "train_list": "",
            "validation_list": "",
            "test_list": "",
            "inference_list": "",
            "replace_final_layer": False,
            "use_mask": False,
            "bidirectional": False,
            "test_images": False,
            "no_verify_image_exists": False
        }

        # Determine the directory of this file
        current_file_dir = Path(__file__).resolve().parent

        # Get the project root
        if current_file_dir.name == "tests":
            project_root = current_file_dir.parent
        else:
            project_root = current_file_dir

        # Add 'src' directory to sys.path
        sys.path.append(str(project_root / 'src'))

        # Set paths for data and model directories
        cls.data_dir = project_root / "tests" / "data"

        cls.sample_image_paths = [os.path.join(
            cls.data_dir, f"test-image{i+1}") for i in range(3)]

        # Extract labels from .txt files
        cls.sample_labels = []
        for i in range(3):
            label_path = os.path.join(cls.data_dir, f"test-image{i+1}.txt")
            with open(label_path, "r") as file:
                cls.sample_labels.append(file.readline().strip())

        # Create sample list file
        cls.sample_list_file = os.path.join(cls.data_dir, "sample_list.txt")
        with open(cls.sample_list_file, "w") as f:
            for img_path, label in zip(cls.sample_image_paths,
                                       cls.sample_labels):
                f.write(f"{img_path}.png\t{label}\n")

        from data.manager import DataManager
        cls.DataManager = DataManager

        from utils.text import Tokenizer
        cls.Tokenizer = Tokenizer

    def _create_temp_file(self, additional_lines=None):
        temp_sample_list_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+")

        for img_path, label in zip(self.sample_image_paths,
                                   self.sample_labels):
            temp_sample_list_file.write(f"{img_path}.png\t{label}\n")
        if additional_lines:
            for line in additional_lines:
                temp_sample_list_file.write(line + "\n")

        temp_sample_list_file.close()
        return temp_sample_list_file.name

    def _remove_temp_file(self, filename):
        os.remove(filename)

    def test_initialization(self):
        # Only provide the required arguments for initialization and check them
        test_config = self.dummy_config.copy()
        test_config.update({
            "batch_size": 32,
            "img_size": (256, 256, 3),
        })

        tokenizer = self.Tokenizer(tokens=list("abc"))
        data_manager = self.DataManager(img_size=test_config["img_size"],
                                        config=test_config,
                                        tokenizer=tokenizer)
        self.assertIsInstance(data_manager, self.DataManager,
                              "DataManager not instantiated correctly")

    def test_create_data_simple(self):
        test_config = self.dummy_config.copy()
        test_config.update({
            "batch_size": 2,
            "img_size": (256, 256, 3),
            "train_list": self.sample_list_file,
            "normalization_file": None,
        })

        data_manager = self.DataManager(img_size=test_config["img_size"],
                                        config=test_config)

        # Check if the data is created correctly
        for i in range(3):
            self.assertEqual(data_manager.get_filename("train", i),
                             self.sample_image_paths[i] + ".png",
                             "Filename not as expected")
            self.assertEqual(data_manager.get_ground_truth("train", i),
                             self.sample_labels[i],
                             "Label not as expected")

        # Batch size is 2, so we should have ceil(3/2) = 2 batches
        self.assertEqual(data_manager.get_train_batches(), 2)

        # Check the tokenizer
        self.assertIsInstance(data_manager.tokenizer, self.Tokenizer,
                              "Tokenizer not created correctly")
        self.assertEqual(len(data_manager.tokenizer), 29,
                         "Charlist length not as expected")

    def test_missing_files(self):
        # Create a temporary list file with a missing image
        temp_sample_list_file = self._create_temp_file(
            additional_lines=[f"missing_image.png\t{self.sample_labels[0]}"])
        test_config = self.dummy_config.copy()
        test_config.update({
            "batch_size": 1,
            "img_size": (256, 256, 3),
            "train_list": temp_sample_list_file,
            "normalization_file": None,
        })

        data_manager = self.DataManager(img_size=test_config["img_size"],
                                        config=test_config)

        # Check if the data is created correctly
        self.assertEqual(data_manager.get_filename("train", 1),
                         self.sample_image_paths[1] + ".png",
                         "Filename not as expected")
        self.assertEqual(data_manager.get_ground_truth("train", 1),
                         self.sample_labels[1],
                         "Label not as expected")

        # The missing image should be ignored, so we should have 3 batches
        self.assertEqual(data_manager.get_train_batches(), 3)

        # Remove the temporary file
        self._remove_temp_file(temp_sample_list_file)

    def test_unsupported_chars_in_eval(self):
        # Create a temporary list file with unsupported characters
        temp_sample_list_file = self._create_temp_file(
            additional_lines=[f"{self.sample_image_paths[0]}.png\t"
                              f"{self.sample_labels[0]}!"])
        test_config = self.dummy_config.copy()
        test_config.update({
            "batch_size": 1,
            "img_size": (256, 256, 3),
            "train_list": temp_sample_list_file,
            "validation_list": temp_sample_list_file,  # Use evaluation list
            "normalization_file": None,
        })

        data_manager = self.DataManager(img_size=test_config["img_size"],
                                        config=test_config)

        # Check if the data is created correctly
        self.assertEqual(data_manager.get_filename("train", 3),
                         self.sample_image_paths[0] + ".png",
                         "Filename not as expected")
        self.assertEqual(data_manager.get_ground_truth("train", 3),
                         self.sample_labels[0]+"!",
                         "Label not as expected")

        # Remove the temporary file
        self._remove_temp_file(temp_sample_list_file)

    def test_injected_charlist(self):
        # Create a temporary list file with unsupported characters
        temp_sample_list_file = self._create_temp_file(
            additional_lines=[f"test-image3.png\t{self.sample_labels[0]}!"])
        test_config = self.dummy_config.copy()
        test_config.update({
            "batch_size": 1,
            "img_size": (256, 256, 3),
            "train_list": temp_sample_list_file,
            "normalization_file": None,
        })
        charlist = list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789, ")
        tokenizer = self.Tokenizer(tokens=charlist)

        data_manager = self.DataManager(img_size=test_config["img_size"],
                                        config=test_config,
                                        tokenizer=tokenizer)

        # Check if the data is created correctly
        self.assertEqual(data_manager.get_filename("train", 2),
                         self.sample_image_paths[2] + ".png",
                         "Filename not as expected")
        self.assertEqual(data_manager.get_ground_truth("train", 2),
                         self.sample_labels[2],
                         "Label not as expected")

        # The label with unsupported characters should be ignored
        self.assertEqual(data_manager.get_train_batches(), 3)

        # Remove the temporary file
        self._remove_temp_file(temp_sample_list_file)

    def test_manager_steps_per_epoch(self):
        # Import the get_arg_parser function
        from setup.arg_parser import get_arg_parser
        from setup.config import Config

        # Create a temporary list file with unsupported characters
        args = []

        parser = get_arg_parser()
        # parser.add_argument('--config_file', type=str, default=None)
        # parser.add_argument('--img_size', type=str, default=None)
        # parser.add_argument('--train_list', type=str, default=None)
        # parser.add_argument('--normalization_file', type=str, default=None)
        # parser.add_argument('--gpu', type=str, default=-1)

        parsed_args = parser.parse_args(args)
        config = Config(args=parsed_args)

        steps_per_epoch = config["steps_per_epoch"]
        # else:
        #     steps_per_epoch = None
        if config["steps_per_epoch"]:
            steps_per_epoch = 5


if __name__ == "__main__":
    unittest.main()
