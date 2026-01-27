import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import unittest
import json
from tempfile import TemporaryDirectory
import logging
import sys
from pathlib import Path

# > Third-party dependencies
import tensorflow as tf


class TestTokenizer(unittest.TestCase):
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

        # Import Tokenizer class
        from utils.text import Tokenizer
        cls.Tokenizer = Tokenizer

    def test_initialize_string_lookup_layers(self):
        # Test initialization with a basic token list
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)

        self.assertEqual(tokenizer.token_list, [
                         '[PAD]', '[UNK]', 'a', 'b', 'c'])
        self.assertIsInstance(tokenizer.token_to_num,
                              tf.keras.layers.StringLookup)
        self.assertIsInstance(tokenizer.num_to_token,
                              tf.keras.layers.StringLookup)

    def test_tokenizer_call(self):
        # Test tokenizing a simple text string
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)

        text = 'abc'
        tokenized_output = tokenizer(text)
        expected_output = [2, 3, 4]  # Corresponding indices of 'a', 'b', 'c'

        self.assertTrue(tf.reduce_all(
            tf.equal(tokenized_output, expected_output)))

    def test_tokenizer_decode(self):
        # Test decoding a sequence of token indices back into text
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)

        tokenized_input = tf.constant([2, 3, 4])  # Indices of 'a', 'b', 'c'
        decoded_text = tokenizer.decode(tokenized_input)

        self.assertEqual(decoded_text, 'abc')

    def test_load_from_file(self):
        # Test loading from a JSON file
        tokens = ['a', 'b', 'c']

        with TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, 'tokenizer.json')
            tokenizer = self.Tokenizer(tokens=tokens)
            tokenizer.save_to_json(json_path)

            loaded_tokenizer = self.Tokenizer.load_from_file(json_path)
            self.assertEqual(loaded_tokenizer.token_list, tokenizer.token_list)

    def test_load_from_legacy_file(self):
        # Test loading from a legacy charlist.txt file and converting to JSON
        chars = ['a', 'b', 'c']
        with TemporaryDirectory() as temp_dir:
            txt_path = os.path.join(temp_dir, 'charlist.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(''.join(chars))

            loaded_tokenizer = self.Tokenizer.load_from_file(txt_path)
            # Skipping [PAD], [UNK]
            self.assertEqual(loaded_tokenizer.token_list[2:], chars)
            self.assertTrue(os.path.exists(
                os.path.join(temp_dir, 'tokenizer.json')))

    def test_save_to_json(self):
        # Test saving tokenizer to a JSON file
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)

        with TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, 'tokenizer.json')
            tokenizer.save_to_json(json_path)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual([data[str(i)]
                             for i in range(len(data))], tokenizer.token_list)

    def test_add_tokens(self):
        # Test adding new tokens
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)

        tokenizer.add_tokens(['d', 'e'])
        self.assertIn('d', tokenizer.token_list)
        self.assertIn('e', tokenizer.token_list)

    def test_empty_token_list(self):
        # Test initializing the tokenizer with an empty token list
        with self.assertRaises(ValueError):
            self.Tokenizer(tokens=[])

    def test_tokenizer_str(self):
        # Test string representation of tokenizer
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)
        tokenizer_str = str(tokenizer)

        expected_str = json.dumps(
            dict(enumerate(tokenizer.token_list)), ensure_ascii=False, indent=4)
        self.assertEqual(tokenizer_str, expected_str)

    def test_tokenizer_len(self):
        # Test length of tokenizer
        tokens = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(tokens=tokens)
        self.assertEqual(len(tokenizer), len(tokenizer.token_list))


if __name__ == '__main__':
    unittest.main()
