import unittest
import numpy as np
from scipy import stats

# Importing the functions from the module
from exercise1 import (
    generate_height_data,
    descriptive_statistics,
    visualise_histogram,
    calculate_percentile,
    identify_outliers,
    random_sampling,
    hypothesis_testing,
    calculate_probability,
)

class TestHeightAnalysis(unittest.TestCase):
    """Unit test class for height data analysis functions.

        This class tests various functions that perform operations on height data,
        such as generating datasets, calculating statistics, identifying outliers,
        and performing hypothesis testing.

        Methods:
            setUp:
                Sets up a dataset with predefined size, mean, and standard deviation
                to be used in the tests.

            test_generate_height_data:
                Tests the function that generates height datasets, ensuring
                the dataset matches the given size, mean, and standard deviation.

            test_descriptive_statistics:
                Tests the calculation of descriptive statistics (mean, standard deviation,
                and median) for a given dataset.

            test_calculate_percentile:
                Verifies the calculation of percentiles (25th, 50th, and 75th)
                and ensures they are in ascending order.

            test_identify_outliers:
                Checks if outliers in the dataset are correctly identified based on
                interquartile range (IQR).

            test_random_sampling:
                Tests random sampling functionality to ensure it selects a
                subset of data correctly.

            test_hypothesis_testing:
                Validates the calculation of the t-statistic and p-value
                for hypothesis testing against a null hypothesis mean.

            test_calculate_probability:
                Tests the calculation of the probability of values exceeding a
                given threshold.

            test_exceptions:
                Ensures that exceptions are raised appropriately for invalid
                inputs across all tested functions.
        """


    def setUp(self):
        """Set up a dataset to test functions."""
        self.size = 1000
        self.mean = 170
        self.std_dev = 10
        self.height_data = generate_height_data(self.size, self.mean, self.std_dev)

    def test_generate_height_data(self):
        """Test dataset generation.

        Verifies the generated dataset matches the specified size, mean, and standard deviation.
        """

        data = generate_height_data(self.size, self.mean, self.std_dev)
        self.assertEqual(len(data), self.size, "Dataset size should match input size.")
        self.assertAlmostEqual(np.mean(data), self.mean, delta=3, msg="Mean should be close to input mean.")
        self.assertAlmostEqual(np.std(data), self.std_dev, delta=3, msg="Standard deviation should match input.")

    def test_descriptive_statistics(self):
        """Test descriptive statistics.

        Ensures mean, standard deviation, and median are calculated correctly.
        """

        mean, std_dev, median = descriptive_statistics(self.height_data)
        self.assertAlmostEqual(mean, self.mean, delta=3, msg="Mean should be close to the dataset mean.")
        self.assertAlmostEqual(std_dev, self.std_dev, delta=3, msg="Std dev should be close to the dataset std dev.")
        self.assertTrue(self.height_data.min() <= median <= self.height_data.max(), "Median should lie within dataset range.")

    def test_calculate_percentile(self):
        """Test percentile calculation.

        Verifies correct calculation of 25th, 50th, and 75th percentiles.
        """

        percentiles = calculate_percentile(self.height_data)
        self.assertEqual(len(percentiles), 3, "Should return three percentiles: 25th, 50th, and 75th.")
        self.assertTrue(percentiles[0] <= percentiles[1] <= percentiles[2], "Percentiles should be in ascending order.")

    def test_identify_outliers(self):
        """Test outlier identification.

        Checks if outliers are identified as values outside the IQR bounds.
        """

        outliers = identify_outliers(self.height_data)
        q1, q3 = np.percentile(self.height_data, [25, 75])
        iqr = q3 - q1
        lower_boundary = q1 - 1.5 * iqr
        upper_boundary = q3 + 1.5 * iqr
        self.assertTrue(all(x < lower_boundary or x > upper_boundary for x in outliers), "Outliers should fall outside IQR bounds.")

    def test_random_sampling(self):
        """Test random sampling.

        Ensures that samples are selected correctly from the original dataset.
        """

        samples = random_sampling(self.height_data)
        self.assertEqual(len(samples), 50, "Should sample 50 elements.")
        self.assertTrue(all(x in self.height_data for x in samples), "All samples should be from the original dataset.")

    def test_hypothesis_testing(self):
        """Test hypothesis testing.

        Verifies the calculation of t-statistic and p-value for a one-sample t-test.
        """

        t_stat, p_value = hypothesis_testing(self.height_data, self.mean)
        self.assertIsInstance(t_stat, float, "t_stat should be a float.")
        self.assertIsInstance(p_value, float, "p_value should be a float.")
        self.assertGreaterEqual(p_value, 0, "p_value should be non-negative.")
        self.assertLessEqual(p_value, 1, "p_value should not exceed 1.")

    def test_calculate_probability(self):
        """Test probability calculation.

        Ensures the correct probability of exceeding a given threshold is calculated.
        """

        probability = calculate_probability(self.height_data, threshold_height=180)
        self.assertGreaterEqual(probability, 0, "Probability should be non-negative.")
        self.assertLessEqual(probability, 1, "Probability should not exceed 1.")
        self.assertIsInstance(probability, float, "Probability should be a float.")

    def test_exceptions(self):
        """Test exceptions for various functions.

        Validates error handling for invalid input parameters.
        """

        # generate_height_data exceptions
        with self.assertRaises(ValueError):
            generate_height_data(size=-10)
        with self.assertRaises(ValueError):
            generate_height_data(size=0)
        with self.assertRaises(TypeError):
            generate_height_data(size="invalid")
        with self.assertRaises(TypeError):
            generate_height_data(mean="invalid")
        with self.assertRaises(ValueError):
            generate_height_data(std_dev=0)
        with self.assertRaises(ValueError):
            generate_height_data(std_dev=-5)

        # descriptive_statistics exceptions
        with self.assertRaises(TypeError):
            descriptive_statistics("invalid")
        with self.assertRaises(ValueError):
            descriptive_statistics(np.array([]))

        # visualise_histogram exceptions
        with self.assertRaises(TypeError):
            visualise_histogram("invalid")
        with self.assertRaises(ValueError):
            visualise_histogram(np.array([]))

        # calculate_percentile exceptions
        with self.assertRaises(TypeError):
            calculate_percentile("invalid")
        with self.assertRaises(ValueError):
            calculate_percentile(np.array([]))

        # identify_outliers exceptions
        with self.assertRaises(TypeError):
            identify_outliers("invalid")
        with self.assertRaises(ValueError):
            identify_outliers(np.array([]))

        # random_sampling exceptions
        with self.assertRaises(TypeError):
            random_sampling("invalid")
        with self.assertRaises(ValueError):
            random_sampling(np.array([1, 2, 3]))

        # hypothesis_testing exceptions
        with self.assertRaises(TypeError):
            hypothesis_testing("invalid")
        with self.assertRaises(ValueError):
            hypothesis_testing(np.array([]))
        with self.assertRaises(TypeError):
            hypothesis_testing(self.height_data, null_hypothesis_mean="invalid")

        # calculate_probability exceptions
        with self.assertRaises(TypeError):
            calculate_probability("invalid")
        with self.assertRaises(ValueError):
            calculate_probability(np.array([]))
        with self.assertRaises(TypeError):
            calculate_probability(self.height_data, threshold_height="invalid")


if __name__ == '__main__':
    unittest.main()
