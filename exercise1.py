import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#https://www.quora.com/How-do-I-generate-a-data-set-with-a-given-mean-a-standard-deviation-and-an-N
def generate_height_data(size = 1000, mean = 170, std_dev = 10):
    """
    Generate a dataset of heights with a given size, mean, and standard deviation.

    Args:
        size (int): Number of data points to generate. Must be a positive integer.
        mean (float): Mean of the dataset.
        std_dev (float): Standard deviation of the dataset. Must be a positive value.

    Returns:
        numpy.ndarray: Array of generated height data.

    Raises:
        TypeError: If size, mean, or std_dev is not of the expected type.
        ValueError: If size is not positive or if std_dev is not positive.
    """

    if not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("Size must be a positive integer.")
    if not isinstance(mean, (int, float)):
        raise TypeError("Mean must be a numeric value.")
    if not isinstance(std_dev, (int, float)) or std_dev <= 0:
        raise ValueError("Standard deviation must be a positive numeric value.")

    dataset = np.random.normal(mean, std_dev, size)
    return dataset

def descriptive_statistics(height_data):
    """
    Calculate descriptive statistics for a dataset.

    Args:
        height_data (numpy.ndarray): Dataset to analyze.

    Returns:
        tuple: Mean, standard deviation, and median of the dataset.

    Raises:
        TypeError: If height_data is not a numpy array.
        ValueError: If height_data is empty.
    """

    if not isinstance(height_data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if height_data.size == 0:
        raise ValueError("Input data cannot be empty.")

    mean = np.mean(height_data)
    std_dev = np.std(height_data)
    median = np.median(height_data)

    return mean, std_dev, median

def visualise_histogram(height_data):
    """
    Visualize a histogram of the dataset.

    Args:
        height_data (numpy.ndarray): Dataset to visualize.

    Raises:
        TypeError: If height_data is not a numpy array.
        ValueError: If height_data is empty.
    """

    if not isinstance(height_data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if height_data.size == 0:
        raise ValueError("Input data cannot be empty.")

    plt.hist(height_data, bins=8, linewidth=0.5, edgecolor="white")
    plt.title("Histogram of Height")
    plt.xlabel("Height [cm]")
    plt.ylabel("Count")
    plt.show()

def calculate_percentile(height_data):
    """
    Calculate the 25th, 50th, and 75th percentiles of the dataset.

    Args:
        height_data (numpy.ndarray): Dataset to analyze.

    Returns:
        numpy.ndarray: Array containing the 25th, 50th, and 75th percentiles.

    Raises:
        TypeError: If height_data is not a numpy array.
        ValueError: If height_data is empty.
    """

    if not isinstance(height_data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if height_data.size == 0:
        raise ValueError("Input data cannot be empty.")


    percentiles = np.percentile(height_data, [25, 50, 75])
    return percentiles

def identify_outliers(height_data):
    """
    Identify outliers in the dataset using the IQR method.

    Args:
        height_data (numpy.ndarray): Dataset to analyze.

    Returns:
        list: List of outlier values.

    Raises:
        TypeError: If height_data is not a numpy array.
        ValueError: If height_data is empty.
    """

    if not isinstance(height_data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if height_data.size == 0:
        raise ValueError("Input data cannot be empty.")

    percentiles = calculate_percentile(height_data)
    q3 = percentiles[2] #75th
    q1 = percentiles[0] #25th

    iqr = q3 - q1

    upper_boundary = q3 + 1.5 * iqr
    lower_boundary = q1 - 1.5 * iqr

    sorted_height_data = sorted(height_data)

    outliers = []

    for data in sorted_height_data:
        if data < lower_boundary or data > upper_boundary:
            outliers.append(data)

    return outliers

#https://numpy.org/doc/2.0/reference/random/generated/numpy.random.choice.html
def random_sampling(height_data):
    """
    Perform random sampling from the dataset.

    Args:
        height_data (numpy.ndarray): Dataset to sample from.

    Returns:
        numpy.ndarray: Array of 50 randomly sampled values.

    Raises:
        TypeError: If height_data is not a numpy array.
        ValueError: If height_data has fewer than 50 elements.
    """

    if not isinstance(height_data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if height_data.size < 50:
        raise ValueError("Input data must have at least 50 elements for sampling.")

    samples = np.random.choice(height_data, 50, replace=False)

    return samples

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
def hypothesis_testing(data, null_hypothesis_mean = 170):
    """
    Perform a one-sample t-test.

    Args:
        data (numpy.ndarray): Dataset to test.
        null_hypothesis_mean (float): Mean under the null hypothesis.

    Returns:
        tuple: t-statistic and p-value from the t-test.

    Raises:
        TypeError: If data is not a numpy array or null_hypothesis_mean is not numeric.
        ValueError: If data is empty.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.size == 0:
        raise ValueError("Input data cannot be empty.")
    if not isinstance(null_hypothesis_mean, (int, float)):
        raise TypeError("Null hypothesis mean must be a numeric value.")

    t_stat, p_value = stats.ttest_1samp(data, null_hypothesis_mean)

    return t_stat, p_value


#https://stackoverflow.com/questions/59333414/finding-the-probability-of-exceeding-certain-threshold
def calculate_probability(data, threshold_height = 180):
    """
    Calculate the probability of exceeding a threshold value in the dataset.

    Args:
        data (numpy.ndarray): Dataset to analyze.
        threshold_height (float): Threshold height value.

    Returns:
        float: Probability of exceeding the threshold.

    Raises:
        TypeError: If data is not a numpy array or threshold_height is not numeric.
        ValueError: If data is empty.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.size == 0:
        raise ValueError("Input data cannot be empty.")
    if not isinstance(threshold_height, (int, float)):
        raise TypeError("Threshold height must be a numeric value.")

    probability = np.sum(data > threshold_height) / data.size

    return probability


if __name__ == '__main__':
    height_data = generate_height_data()

    print("Statistics:")
    print("Mean: ", descriptive_statistics(height_data)[0])
    print("Standard Deviation: ", descriptive_statistics(height_data)[1])
    print("Median: ", descriptive_statistics(height_data)[2])

    visualise_histogram(height_data)

    print("\nPercentiles:")
    print("25th percentile: ", calculate_percentile(height_data)[0])
    print("50th percentile: ", calculate_percentile(height_data)[1])
    print("75th percentile: ", calculate_percentile(height_data)[2])

    print("\nOutliers:")
    print(identify_outliers(height_data))

    print("\nRandom Sampling:")
    print(random_sampling(height_data))

    print("\nHypothesis Testing:")
    print("T-statistic: ", hypothesis_testing(height_data)[0])
    print("Probability-value: ", calculate_probability(height_data)[1])

    print("\nCalculate Probability:")
    print(calculate_probability(height_data))


