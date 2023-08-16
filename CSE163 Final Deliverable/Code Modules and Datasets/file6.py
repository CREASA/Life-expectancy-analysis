"""
Calls the test functions to test the
functionality of the clean_data, plot_distributions,
no_outliers_plot_distribution, create_scatter_plot,
create_country_scatter_plots,
and create_country_scatter_plot_to functions in their
respective files.
"""
from file1 import clean_data
from file2 import plot_distributions, no_outliers_plot_distribution
from file5 import create_country_scatter_plot_to
from file5 import create_scatter_plot
import pandas as pd
import numpy as np


def test_clean_data() -> None:
    """
    Tests the clean_data function in file1.py by
    verifying that the function cleans the data
    and creates a clean_data.csv file with no NaN values.
    :return: None.
    """
    clean_data()
    data = pd.read_csv("clean_data.csv")
    num_nans = data.isna().sum().sum()
    assert data.shape == (4522, 6)
    assert num_nans == 0


def test_functions() -> None:
    """
    Tests the plot_distributions, no_outliers_plot_distribution
    functions in file2.py
    by verifying that they produce the expected output given
    test data.
    :return: None.
    """
    # load data
    import seaborn as sns
    data = sns.load_dataset('anscombe')
    data.rename(columns={'y': 'LifeExp', 'x': 'GDP'}, inplace=True)
    data['CO2'] = np.random.normal(loc=20, scale=5, size=data.shape[0])
    data['Health_Expenditure'] = np.random.normal(loc=2000,
                                                  scale=500,
                                                  size=data.shape[0])

    # test plot_distributions function
    plot_distributions(data)

    # test plot_distribution function with LifeExp column
    no_outliers_plot_distribution(data, "LifeExp")

    # test plot_distribution function with Health_Expenditure column
    no_outliers_plot_distribution(data, "Health_Expenditure")


def test_create_scatter_plot() -> None:
    """
    Tests the create_scatter_plot function in file4.py by
    verifying that it produces the expected output
    given test data.
    :return: None.
    """
    data = pd.DataFrame({'Health_Expenditure': [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                         'LifeExp': [76.5, 77.0, 77.5, 78.0, 78.5, 79.0]})
    create_scatter_plot(data, 'Health_Expenditure', 'LifeExp')


def test_create_country_scatter_plot_to() -> None:
    """
    Tests the create_country_scatter_plot_to
    function in file4.py by verifying that it
    produces the expected output
    given test data.
    :return: None.
    """
    data = pd.read_csv("test_dataset.csv")
    create_country_scatter_plot_to(data)


def main():
    test_clean_data()
    test_functions()
    test_create_scatter_plot()
    test_create_country_scatter_plot_to()


if __name__ == "__main__":
    main()
