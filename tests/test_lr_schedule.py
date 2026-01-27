import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import logging
import unittest
from pathlib import Path
import sys

# > Third-party dependencies
import tensorflow as tf


class TestLoghiLearningRateSchedule(unittest.TestCase):
    """
    Test the LoghiLearningRateSchedule class.

    Test coverage:
    1. `test_initialization`: Test correct initialization.
    2. `test_invalid_lr_init`: Test invalid initialization arguments.
    3. `test_invalid_decay_rate_init`: Test invalid initialization arguments
       related to the decay rate.
    4. `test_invalid_decay_steps_init`: Test invalid initialization arguments
       related to the decay steps.
    5. `test_invalid_warmup_ratio_init`: Test invalid initialization arguments
       related to the warmup ratio.
    6. `test_invalid_total_steps_init`: Test invalid initialization arguments
       related to the total steps.
    7. `test_call_method`: Test the __call__ method for various scenarios.
    8. `test_get_config`: Test the get_config method.
    9. `test_learning_rate_calculation`: Test learning rate at various steps.
    10. `test_exponential_decay_compatibility_stepwise`: Test compatibility
        with TensorFlow's ExponentialDecay schedule (stepwise decay).
    11. `test_exponential_decay_compatibility_epochwise`: Test compatibility
        with TensorFlow's ExponentialDecay schedule (epochwise decay).
    """

    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR
        )

        # Determine the directory of this file
        current_file_dir = Path(__file__).resolve().parent

        # Get the project root
        if current_file_dir.name == "tests":
            project_root = current_file_dir.parent
        else:
            project_root = current_file_dir

        # Add 'src' directory to sys.path
        sys.path.append(str(project_root / 'src'))

        # Import the LoghiLearningRateSchedule class
        from model.optimization import LoghiLearningRateSchedule
        cls.LoghiLearningRateSchedule = LoghiLearningRateSchedule

    def setUp(self):
        # Common setup for all tests
        self.initial_learning_rate = 0.1
        self.decay_rate = 0.96
        self.decay_steps = 1000
        self.warmup_ratio = 0.1
        self.total_steps = 10000
        self.lr_schedule = self.LoghiLearningRateSchedule(
            initial_learning_rate=self.initial_learning_rate,
            decay_rate=self.decay_rate,
            decay_steps=self.decay_steps,
            warmup_ratio=self.warmup_ratio,
            total_steps=self.total_steps
        )

    def test_initialization(self):
        # Test correct initialization
        self.assertAlmostEqual(self.lr_schedule.initial_learning_rate,
                               self.initial_learning_rate)
        self.assertAlmostEqual(self.lr_schedule.decay_rate, self.decay_rate)
        self.assertEqual(self.lr_schedule.decay_steps, self.decay_steps)
        self.assertEqual(self.lr_schedule.warmup_steps,
                         self.warmup_ratio * self.total_steps)
        self.assertAlmostEqual(
            self.lr_schedule.warmup_ratio, self.warmup_ratio)
        self.assertEqual(self.lr_schedule.total_steps, self.total_steps)
        self.assertEqual(self.lr_schedule.decay_per_epoch, False)
        self.assertEqual(self.lr_schedule.linear_decay, False)

    def test_invalid_lr_init(self):
        # Test invalid initialization arguments
        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=-1,
                decay_rate=self.decay_rate,
                decay_steps=self.decay_steps,
                warmup_ratio=self.warmup_ratio,
                total_steps=self.total_steps
            )
        self.assertIn("Initial learning rate must be positive",
                      str(context.exception))

    def test_invalid_decay_rate_init(self):
        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=-1,
                decay_steps=self.decay_steps,
                warmup_ratio=self.warmup_ratio,
                total_steps=self.total_steps
            )
        self.assertIn("Decay rate must be positive", str(context.exception))

    def test_invalid_decay_steps_init(self):
        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=self.decay_rate,
                decay_steps=-1,
                warmup_ratio=self.warmup_ratio,
                total_steps=self.total_steps
            )
        self.assertIn("Decay steps must be a non-negative integer",
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=self.decay_rate,
                decay_steps=100.1,
                warmup_ratio=self.warmup_ratio,
                total_steps=self.total_steps
            )
        self.assertIn("Decay steps must be a non-negative integer",
                      str(context.exception))

    def test_invalid_warmup_ratio_init(self):
        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=self.decay_rate,
                decay_steps=self.decay_steps,
                warmup_ratio=-1,
                total_steps=self.total_steps
            )
        self.assertIn("Warmup ratio must be between 0 and 1",
                      str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=self.decay_rate,
                decay_steps=self.decay_steps,
                warmup_ratio=1.1,
                total_steps=self.total_steps
            )
        self.assertIn("Warmup ratio must be between 0 and 1",
                      str(context.exception))

    def test_invalid_total_steps_init(self):
        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=self.decay_rate,
                decay_steps=self.decay_steps,
                warmup_ratio=self.warmup_ratio,
                total_steps=-1
            )
        self.assertIn("Total steps must be a positive integer",
                      str(context.exception))
        with self.assertRaises(ValueError) as context:
            self.LoghiLearningRateSchedule(
                initial_learning_rate=self.initial_learning_rate,
                decay_rate=self.decay_rate,
                decay_steps=self.decay_steps,
                warmup_ratio=self.warmup_ratio,
                total_steps=100.1
            )
        self.assertIn("Total steps must be a positive integer",
                      str(context.exception))

    def test_call_method(self):
        # Test the __call__ method for various scenarios
        warmup_end_step = int(self.total_steps * self.warmup_ratio)
        post_warmup_step = warmup_end_step + 1
        mid_decay_step = self.total_steps // 2
        final_step = self.total_steps

        # During warmup
        warmup_lr = self.lr_schedule(tf.constant(warmup_end_step - 1))
        self.assertAlmostEqual(warmup_lr, self.initial_learning_rate *
                               ((warmup_end_step - 1) / self.decay_steps),
                               places=5)

        # Just after warmup
        post_warmup_lr = self.lr_schedule(tf.constant(post_warmup_step))
        self.assertNotEqual(post_warmup_lr, warmup_lr)

        # Middle of decay phase
        mid_decay_lr = self.lr_schedule(tf.constant(mid_decay_step))
        self.assertLess(mid_decay_lr, self.initial_learning_rate)

        # Final step
        final_lr = self.lr_schedule(tf.constant(final_step))
        self.assertLess(final_lr, mid_decay_lr)

    def test_get_config(self):
        # Test the get_config method
        config = self.lr_schedule.get_config()
        self.assertAlmostEqual(config["initial_learning_rate"],
                               self.initial_learning_rate)
        self.assertAlmostEqual(config["decay_rate"], self.decay_rate)
        self.assertEqual(config["decay_steps"], self.decay_steps)
        self.assertEqual(config["total_steps"], self.total_steps)
        self.assertEqual(config["linear_decay"], False)

    def test_learning_rate_calculation(self):
        # Test learning rate at various steps
        for step in [0, int(self.total_steps * self.warmup_ratio),
                     self.decay_steps, self.total_steps]:
            lr = self.lr_schedule(tf.constant(step))
            self.assertIsInstance(lr, tf.Tensor)

    def test_exponential_decay_compatibility_stepwise(self):
        total_steps = 10000
        initial_learning_rate = 0.1
        decay_rate = 0.96
        decay_steps = 1000

        # LoghiLearningRateSchedule setup
        loghi_lr_schedule = self.LoghiLearningRateSchedule(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            warmup_ratio=0,  # No warmup
            total_steps=total_steps,
            linear_decay=False  # Exponential decay
        )

        # TensorFlow ExponentialDecay setup
        tf_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )

        for step in range(total_steps):
            loghi_lr = loghi_lr_schedule(step)
            tf_lr = tf_lr_schedule(step)
            self.assertAlmostEqual(
                loghi_lr, tf_lr, places=5,
                msg=f"Learning rates differ at step {step}")

    def test_exponential_decay_compatibility_epochwise(self):
        total_steps = 10000
        initial_learning_rate = 0.1
        decay_rate = 0.96
        decay_steps = 1000

        # LoghiLearningRateSchedule setup
        loghi_lr_schedule = self.LoghiLearningRateSchedule(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            warmup_ratio=0,  # No warmup
            total_steps=total_steps,
            linear_decay=False,  # Exponential decay
            decay_per_epoch=True
        )

        # TensorFlow ExponentialDecay setup
        tf_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        for step in range(total_steps):
            loghi_lr = loghi_lr_schedule(step)
            tf_lr = tf_lr_schedule(step)
            self.assertAlmostEqual(
                loghi_lr, tf_lr, places=5,
                msg=f"Learning rates differ at step {step}")


if __name__ == "__main__":
    unittest.main()
