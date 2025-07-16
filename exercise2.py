import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """
    Load the heart disease dataset from a CSV file.

    Returns
    -------
    pandas.DataFrame or None
        The loaded dataset as a pandas DataFrame if successful, or None if the file is not found.

    Preconditions
    -------------
    The file 'heart_disease_dataset.csv' must exist in the same directory as this script.

    Postconditions
    --------------
    If the file is found, a DataFrame is returned with the contents of the CSV file.

    Raises
    ------
    FileNotFoundError
        If the file 'heart_disease_dataset.csv' does not exist.
    """

    try:
        heart_disease_data = pd.read_csv('heart_disease_dataset.csv')
        return heart_disease_data
    except FileNotFoundError as e:
        print("Error: The file 'heart_disease_dataset.csv' was not found.")
        raise e

def who_suffers_more_from_disease(heart_disease_data):
    """
    Determine whether men or women suffer more from heart disease and by what percentage.

    Parameters
    ----------
    heart_disease_data : pandas.DataFrame
        The dataset containing heart disease data, including columns 'Sex' and 'Disease'.

    Returns
    -------
    tuple
        A tuple containing the gender ('men' or 'women') that suffers more from heart disease and the percentage difference.

    Preconditions
    -------------
    The DataFrame must contain columns 'Sex' and 'Disease', and 'Disease' must be boolean.

    Postconditions
    --------------
    A tuple with the gender suffering more and the percentage difference is returned.

    Raises
    ------
    ValueError
        If the required columns ('Sex', 'Disease') are missing.
    TypeError
        If 'Disease' column is not of boolean type.
    """

    if 'Sex' not in heart_disease_data.columns or 'Disease' not in heart_disease_data.columns:
        raise ValueError("The dataset must contain 'Sex' and 'Disease' columns.")

    if not pd.api.types.is_bool_dtype(heart_disease_data['Disease']):
        raise TypeError("The 'Disease' column must be of boolean type.")

    # Filter rows for males with disease and calculate the count
    men_with_disease = heart_disease_data.loc[(heart_disease_data['Sex'] == 'male') & (heart_disease_data['Disease'] == True)].shape[0]  # Count rows

    # Filter rows for females with disease and calculate the count
    women_with_disease = heart_disease_data.loc[(heart_disease_data['Sex'] == 'female') & (heart_disease_data['Disease'] == True)].shape[0]  # Count rows


    higher = max(men_with_disease, women_with_disease)
    lower = min(men_with_disease, women_with_disease)
    percentage_difference = (higher - lower) / higher * 100

    if men_with_disease > women_with_disease:
        return 'men',percentage_difference
    else:
        return 'women',percentage_difference


def average_value_of_serum_cholesterol(heart_disease_data):
    """
    Calculate the average serum cholesterol levels grouped by sex and disease presence.

    Parameters
    ----------
    heart_disease_data : pandas.DataFrame
        The dataset containing heart disease data, including columns 'Sex', 'Disease', and 'Serum cholesterol in mg/dl'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame showing the average serum cholesterol levels grouped by sex and disease presence.

    Preconditions
    -------------
    The DataFrame must contain 'Sex', 'Disease', and 'Serum cholesterol in mg/dl' columns.

    Postconditions
    --------------
    A new DataFrame is returned with grouped averages.

    Raises
    ------
    ValueError
        If the required columns ('Sex', 'Disease', 'Serum cholesterol in mg/dl') are missing.
    """

    required_columns = ['Sex', 'Disease', 'Serum cholesterol in mg/dl']
    for column in required_columns:
        if column not in heart_disease_data.columns:
            raise ValueError(f"The dataset must contain the column '{column}'.")

    avg_cholesterol = heart_disease_data.groupby(['Sex', 'Disease'])['Serum cholesterol in mg/dl'].mean()
    avg_cholesterol_df = avg_cholesterol.reset_index()  # Convert to DataFrame for better readability
    print("\nAverage Serum Cholesterol Levels by Sex and Disease:")
    print(avg_cholesterol_df)
    return avg_cholesterol_df


def visualise_histogram(heart_disease_data):
    """
    Visualize the age distribution of patients with heart disease using a histogram.

    Parameters
    ----------
    heart_disease_data : pandas.DataFrame
        The dataset containing heart disease data, including columns 'Age' and 'Disease'.

    Preconditions
    -------------
    The DataFrame must contain the 'Age' and 'Disease' columns.

    Postconditions
    --------------
    A histogram is displayed showing the age distribution of patients with heart disease.

    Raises
    ------
    ValueError
        If the required columns ('Age', 'Disease') are missing.
    """

    if 'Age' not in heart_disease_data.columns or 'Disease' not in heart_disease_data.columns:
        raise ValueError("The dataset must contain 'Age' and 'Disease' columns.")

    ages_with_disease = heart_disease_data[heart_disease_data['Disease'] == True]['Age']
    plt.title('Age Distribution of Patients with Heart Disease')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.hist(ages_with_disease, bins=10, edgecolor='black')
    plt.show()

def visualise_boxplot(heart_disease_data):
    """
    Create a boxplot to visualize maximum heart rate achieved by disease presence.

    Parameters
    ----------
    heart_disease_data : pandas.DataFrame
        The dataset containing heart disease data, including columns 'Maximum heart rate achieved' and 'Disease'.

    Preconditions
    -------------
    The DataFrame must contain 'Maximum heart rate achieved' and 'Disease' columns.

    Postconditions
    --------------
    A boxplot is displayed comparing maximum heart rate by disease presence.

    Raises
    ------
    ValueError
        If the required columns ('Maximum heart rate achieved', 'Disease') are missing.
    """

    required_columns = ['Maximum heart rate achieved', 'Disease']
    for column in required_columns:
        if column not in heart_disease_data.columns:
            raise ValueError(f"The dataset must contain the column '{column}'.")

    data_to_plot = [
        heart_disease_data[heart_disease_data['Disease'] == False]['Maximum heart rate achieved'],
        heart_disease_data[heart_disease_data['Disease'] == True]['Maximum heart rate achieved']
    ]

    # Create the box plot
    plt.boxplot(data_to_plot, tick_labels=['No Disease', 'Disease'], patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='black'), medianprops=dict(color='red'))

    # Customize the plot
    plt.title('Maximum Heart Rate Achieved by Disease Presence')
    plt.xlabel('Disease Presence')
    plt.ylabel('Maximum Heart Rate Achieved (bpm)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability
    plt.show()

def visualise_barchart(heart_disease_data):
    """
    Create a bar chart showing the frequency of exercise-induced angina by disease presence.

    Parameters
    ----------
    heart_disease_data : pandas.DataFrame
        The dataset containing heart disease data, including columns 'Exercise induced angina' and 'Disease'.

    Preconditions
    -------------
    The DataFrame must contain 'Exercise induced angina' and 'Disease' columns.

    Postconditions
    --------------
    A bar chart is displayed showing the frequency of exercise-induced angina by disease presence.

    Raises
    ------
    ValueError
        If the required columns ('Exercise induced angina', 'Disease') are missing.
    """

    if 'Exercise induced angina' not in heart_disease_data.columns or 'Disease' not in heart_disease_data.columns:
        raise ValueError("The dataset must contain 'Exercise induced angina' and 'Disease' columns.")

    frequency_of_disease = heart_disease_data.groupby(['Exercise induced angina', 'Disease']).size().unstack(fill_value=0)
    frequency_of_disease.plot(kind='bar', edgecolor='black', color=['skyblue', 'orange'])
    plt.title('Frequency of Exercise Induced angina')
    plt.xlabel('Exercise induced angina')
    plt.ylabel('Frequency')
    plt.legend(title='Disease', labels=['No Disease', 'Disease'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    heart_disease_data = load_data()
    if heart_disease_data is not None:
        gender, percentage = who_suffers_more_from_disease(heart_disease_data)
        print(f"{gender.capitalize()} suffer more from heart disease by {percentage:.2f}%.\n")

        avg_cholesterol_df = average_value_of_serum_cholesterol(heart_disease_data)

        visualise_histogram(heart_disease_data)
        visualise_boxplot(heart_disease_data)
        visualise_barchart(heart_disease_data)

