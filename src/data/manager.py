# Imports

# > Standard library
from collections import defaultdict
import logging
import os
from typing import Dict, Tuple, Optional, List, Set

# > Third party dependencies
import tensorflow as tf
import numpy as np

# > Local dependencies
from data.loader import DataLoader
from setup.config import Config
from utils.text import Tokenizer, normalize_text
from bidi.algorithm import get_display

# Safe bidi wrapper
CONTROL_CHARS = {"\u202A", "\u202B", "\u202C", "\u202D", "\u202E", "\u200E", "\u200F", "\u2066", "\u2067", "\u2068", "\u2069", "\uFEFF"}
RTL_BLOCK_RANGES = [(0x0590, 0x08FF)]  # Broad Hebrew+Arabic range

def _contains_rtl(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for start, end in RTL_BLOCK_RANGES:
            if start <= cp <= end:
                return True
    return False

def _sanitize_bidi_text(text: str) -> str:
    cleaned = ''.join(ch for ch in text if ch not in CONTROL_CHARS)
    # Keep printable only (avoid triggering bidi assertions on weird codepoints)
    cleaned = ''.join(ch for ch in cleaned if ch.isprintable())
    return cleaned or text

def safe_get_display(text: str) -> str:
    text = _sanitize_bidi_text(text)
    if not _contains_rtl(text):
        return text
    try:
        return get_display(text)
    except AssertionError:
        logging.debug("safe_get_display: AssertionError on text, returning sanitized original.")
        return text
    except Exception as e:
        logging.debug("safe_get_display: Exception %s on text, returning sanitized original.", e)
        return text


class DataManager:
    """
    Class for creating and managing datasets for training, validating, etc.

    Parameters
    ----------
    img_size : Tuple[int, int, int]
        The size of the input images (height, width, channels).
    config : Config
        The configuration dictionary containing various settings.
    tokenizer : Optional[Tokenizer], optional
        The tokenizer to use for encoding text data, by default None.
    """

    def __init__(self,
                 img_size: Tuple[int, int, int],
                 config: Config,
                 tokenizer: Optional[Tokenizer] = None,
                 augmentation_model: Optional[tf.keras.Model] = None):

        self.height = img_size[0]
        self.channels = img_size[2]
        self.config = config
        self.tokenizer = tokenizer
        self.augmentation_model = augmentation_model

        # Determine the evaluation list
        self.evaluation_list = None
        if self.config['train_list'] and self.config['validation_list']:
            self.evaluation_list = self.config['validation_list']

        # Process the raw data and create file names, labels, sample weights,
        # and tokenizer
        logging.info("Processing raw data...")
        file_names, labels, sample_weights, self.tokenizer = self._process_raw_data()

        # Initialize raw_data dictionary for each dataset split
        self.raw_data = {}
        for split in ['train', 'evaluation', 'validation', 'test', 'inference']:
            self.raw_data[split] = (
                file_names[split],  # File names for the split
                labels[split],  # Labels for the split
                sample_weights[split]  # Sample weights for the split
            )

        # Fill the datasets dictionary with datasets for different partitions
        logging.info("Creating datasets...")
        try:
            steps_per_epoch = config["steps_per_epoch"]
        except KeyError:
            steps_per_epoch = None
        self.datasets = self._fill_datasets_dict(file_names, labels,
                                                 sample_weights, steps_per_epoch)

    def _process_raw_data(self) -> Tuple[Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Tokenizer]:
        """
        Process the raw data and create file names, labels, sample weights,
        and tokenizer.

        Returns
        -------
        Tuple[Dict[str, List[str]],
              Dict[str, List[str]],
              Dict[str, List[str]],
              Tokenizer]
            A tuple containing dictionaries of file names, labels, sample
            weights, and the tokenizer.
        """

        # Initialize data partitions with corresponding labels
        file_names_dict = defaultdict(list)
        labels_dict = defaultdict(list)
        sample_weights_dict = defaultdict(list)
        characters = set()
        if self.tokenizer and not self.config["replace_final_layer"]:
            characters = set(self.tokenizer.token_list)
        else:
            self.tokenizer = None

        for partition in ('train', 'evaluation', 'validation',
                          'test', 'inference'):
            partition_text_file = self.config[f"{partition}_list"] \
                if partition != "evaluation" else self.evaluation_list

            if partition_text_file:
                # Create data for the current partition
                file_names, labels, sample_weights = self._create_data(
                    partition_name=partition,
                    text_file=partition_text_file,
                    characters=characters,
                    bidirectional=self.config["bidirectional"],
                    test_images=self.config["test_images"],
                    no_verify_image_exists=self.config["no_verify_image_exists"]
                )
                if len(file_names) == 0:
                    raise ValueError("No data found for the specified "
                                     f"{partition} list(s). Have you verified "
                                     "the data paths?")

                # Fill the dictionary with the data
                file_names_dict[partition] = file_names
                labels_dict[partition] = labels
                sample_weights_dict[partition] = sample_weights

        # Initialize the tokenizer if not provided
        if self.tokenizer is None:
            if not characters:
                raise ValueError(
                    "Character list is empty after creating training data.")
            self.tokenizer = Tokenizer(sorted(characters))
        return file_names_dict, labels_dict, sample_weights_dict, self.tokenizer

    def _fill_datasets_dict(self,
                            partitions: Dict[str, List[str]],
                            labels: Dict[str, List[str]],
                            sample_weights: Dict[str, List[str]],
                            steps_per_epoch: int) \
            -> Dict[str, tf.data.Dataset]:
        """
        Initialize data generators for different dataset partitions and
        update tokenizer based on the dataset.

        Parameters
        ----------
        partitions : Dict[str, List[str]]
            A dictionary containing lists of file names for each partition.
        labels : Dict[str, List[str]]
            A dictionary containing lists of labels for each partition.
        sample_weights : Dict[str, List[str]]
            A dictionary containing lists of sample weights for each partition.

        Returns
        -------
        Dict[str, tf.data.Dataset]
            A dictionary containing datasets for each partition.
        """

        # Create datasets for different partitions
        datasets = defaultdict(lambda: None)

        for partition in ('train', 'evaluation', 'validation',
                          'test', 'inference'):
            # Special case for evaluation partition, since there is no
            # evaluation_list in the config, but the evaluation_list is
            # inferred from the validation_list and train_list in the init
            if partition == "evaluation":
                partition_list = self.evaluation_list
            else:
                partition_list = self.config[f"{partition}_list"]
            if partition_list:
                # Create dataset for the current partition
                datasets[partition] = self._create_dataset(
                    files=partitions[partition],
                    labels=labels[partition],
                    sample_weights=sample_weights[partition],
                    partition_name=partition,
                    steps_per_epoch=steps_per_epoch
                )

        return datasets

    def _create_data(self,
                     partition_name: str,
                     text_file: str,
                     characters: Set[str],
                     bidirectional: bool,
                     test_images: bool,
                     no_verify_image_exists: bool) -> Tuple[List[str], List[str], List[str]]:
        """
        Create data for a specific partition from a text file.

        Parameters
        ----------
        partition_name : str
            The name of the partition.
        text_file : str
            The path to the text file containing the data.
        characters : Set[str]
            The set of characters to be used for the tokenizer.
        bidirectional : bool
            Whether to use bidirectional text processing.

        Returns
        -------
        Tuple[List[str], List[str], List[str]]
            A tuple containing lists of file names, labels, and sample weights.
        """

        # Define the lists for the current partition
        labels, partitions, sample_weights = [], [], []

        # Define the faulty lines and flaws
        faulty_lines = {}
        flaw_counts = {}

        # Process each file in the data files list
        for file_path in text_file.split(' '):
            file_path = file_path.strip()
            if not file_path:
                continue
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist")

            with open(file_path, encoding="utf-8") as file:
                # Iterate over the lines in the file
                for line in file:
                    data, flaw = self._process_line(line,
                                                    partition_name,
                                                    characters,
                                                    test_images,
                                                    no_verify_image_exists)
                    if data is not None:
                        file_name, ground_truth, sample_weight = data
                        partitions.append(file_name)
                        if bidirectional:
                            if getattr(self.config.args, 'safe_bidi', False):
                                ground_truth = safe_get_display(ground_truth)
                            else:
                                try:
                                    ground_truth = get_display(ground_truth)
                                except AssertionError:
                                    logging.warning("python-bidi assertion encountered; falling back to safe_get_display. Consider using --safe_bidi to suppress this warning.")
                                    ground_truth = safe_get_display(ground_truth)
                        labels.append(ground_truth)
                        sample_weights.append(str(sample_weight))
                    else:
                        faulty_lines[line] = flaw
                        flaw_counts[flaw] = flaw_counts.get(flaw, 0) + 1

        # Log the faulty lines and flaw counts
        if faulty_lines:
            logging.warning("Faulty lines for %s:", partition_name)
            # Sort the faulty lines by flaw
            for line, flaw in faulty_lines.items():
                if "\u009D" in line:
                    line = line.replace("\u009D", "")
                    logging.warning("Special character found in line: %s", line)
                # Remove the special character that is causing the output to stop

                logging.warning("%s: %s", flaw, line)

            logging.warning("Flaw counts for %s:", partition_name)
            for flaw, count in flaw_counts.items():
                logging.warning("%s: %d", flaw, count)

        logging.info("Created data for %s with %d samples",
                     partition_name, len(partitions))

        return partitions, labels, sample_weights

    def _process_line(self,
                      line: str,
                      partition_name: str,
                      characters: Set[str],
                      test_images: bool,
                      no_verify_image_exists: bool) \
            -> Tuple[Optional[Tuple[str, str, float]], Optional[str]]:
        """
        Process a single line from the data file.

        Parameters
        ----------
        line : str
            The line to process.
        partition_name : str
            The name of the partition.
        characters : Set[str]
            The set of characters to use for validation.

        Returns
        -------
        Tuple[Optional[Tuple[str, str, float]], Optional[str]]
            A tuple containing the processed data (file name, ground truth,
            sample weight) and the flaw (if any).
        """

        # Strip the line of leading and trailing whitespace
        line = line.strip()

        # Skip empty and commented lines
        if not line or line.startswith('#'):
            return None, "Empty or commented line"

        # Split the line into tab-separated fields:
        # filename sample_weight ground_truth
        fields = line.split('\t')

        # Extract the filename and ground truth from the fields
        file_name = fields[0]

        # Skip missing files
        if not no_verify_image_exists and not os.path.exists(file_name):
            logging.warning("Missing: %s in %s. Skipping...",
                            file_name, partition_name)
            return None, "Missing file"
        if test_images:
        #     load image
            # Check if the file is a valid image
            try:
                image = tf.io.read_file(file_name)
                image = tf.image.decode_image(image, channels=self.channels)
            except Exception as e:
                logging.warning("Invalid image file: %s. Skipping...", file_name)
                return None, "Invalid image file"

        # Extract the ground truth from the fields
        ground_truth, flaw = self._get_ground_truth(fields, partition_name)
        if ground_truth is None:
            return None, flaw

        # Normalize the ground truth if a normalization file is provided and
        # the partition is either 'train' or 'evaluation'
        if self.config['normalization_file'] and \
                partition_name in ('train', 'evaluation'):
            ground_truth = normalize_text(ground_truth,
                                          self.config['normalization_file'])

        # Check for unsupported characters in the ground truth
        if not self._is_valid_ground_truth(ground_truth, partition_name,
                                           characters):
            return None, "Unsupported characters"

        sample_weight = self._get_sample_weight(fields)

        return (file_name, ground_truth, sample_weight), None

    def _get_ground_truth(self,
                          fields: List[str],
                          partition_name: str) \
            -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the ground truth from the fields.

        Parameters
        ----------
        fields : List[str]
            The fields from the line.
        partition_name : str
            The name of the partition.

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            A tuple containing the ground truth and the flaw (if any).
        """

        # Collect the ground truth and skip lines with empty ground truth
        # unless it's an inference partition
        ground_truth = fields[-1] if len(fields) > 1 else ""

        if not ground_truth:
            if partition_name == "inference":
                ground_truth = "to be determined"
            else:
                return None, "Empty ground truth"

        return ground_truth, None

    def _is_valid_ground_truth(self,
                               ground_truth: str,
                               partition_name: str,
                               characters: Set[str]) -> bool:
        """
        Check if the ground truth is valid.

        Parameters
        ----------
        ground_truth : str
            The ground truth to check.
        partition_name : str
            The name of the partition.
        characters : Set[str]
            The set of characters.

        Returns
        -------
        bool
            True if the ground truth is valid, False otherwise.
        """

        # Check for unsupported characters in the ground truth
        unsupported_characters = set(ground_truth) - characters
        if unsupported_characters:
            # Unsupported characters are allowed in the validation, inference,
            # and test partitions, but not in the evaluation partition
            if partition_name in ('validation', 'inference', 'test'):
                return True

            if partition_name == 'train' and not self.tokenizer:
                # Add unsupported characters if we don't have a tokenizer yet
                # or if we're allowed to replace the final layer
                characters.update(unsupported_characters)
                return True

            return False
        return True

    def _get_sample_weight(self,
                           fields: List[str]) -> float:
        """
        Extract the sample weight from the fields.

        Parameters
        ----------
        fields : List[str]
            The fields from the line.

        Returns
        -------
        float
            The sample weight.
        """

        # Extract the sample weight from the fields
        sample_weight = 1.0
        if len(fields) > 2:
            try:
                sample_weight = float(fields[1])
            except ValueError:
                pass
        return sample_weight

    def get_filename(self, partition: str, item_id: int):
        """ Get the filename for the given partition and item id """
        return self.raw_data[partition][0][item_id]

    def get_ground_truth(self, partition: str, item_id: int):
        """ Get the ground truth for the given partition and item id """
        return self.raw_data[partition][1][item_id]

    def get_train_batches(self):
        """ Get the number of batches for training """
        total = len(self.raw_data['train'][0])
        return int(np.ceil(total / self.config['batch_size']))

    def _create_dataset(self,
                        files: List[str],
                        labels: List[str],
                        sample_weights: List[str],
                        partition_name: str,
                        steps_per_epoch: int) -> tf.data.Dataset:
        """
        Create a dataset for a specific partition.

        Parameters
        ----------
        files : List[str]
            A list of file names.
        labels : List[str]
            A list of labels.
        sample_weights : List[str]
            A list of sample weights.
        partition_name : str
            The name of the partition.

        Returns
        -------
        tf.data.Dataset
            The created dataset.
        """
        is_training = partition_name == 'train'

        # Zip the datasets together
        dataset = tf.data.Dataset.from_tensor_slices(
            list(zip(files, labels, sample_weights))
        )

        # Shuffle and repeat if training
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(files),
                                      reshuffle_each_iteration=True)
            dataset = dataset.repeat()

        # Create the data loader
        data_loader = DataLoader(
            tokenizer=self.tokenizer,
            height=self.height,
            channels=self.channels,
            augmentation_model=self.augmentation_model,
            is_training=is_training
        )

        # Map the processing function with parallel calls
        dataset = dataset.map(
            lambda inputs: data_loader.process_sample(inputs),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )

        # Batch the dataset
        dataset = dataset.padded_batch(
            batch_size=self.config["batch_size"],
            padded_shapes=(
                [None, None, self.channels],  # Image shape
                [None],  # Label shape
                []  # Sample weight shape
            ),
            padding_values=(
                tf.constant(-10, dtype=tf.float32),  # Image padding value
                tf.constant(0, dtype=tf.int64),  # Label padding value
                tf.constant(1.0, dtype=tf.float32)  # Sample weight padding value
            ),
            drop_remainder=False  # Keep the last batch even if it's smaller
        )

        # Prefetch data
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Assert the cardinality of the dataset if training
        if steps_per_epoch is None:
            batches = int(np.ceil(len(self.raw_data[partition_name][0]) / self.config['batch_size']))
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(batches))

        return dataset
