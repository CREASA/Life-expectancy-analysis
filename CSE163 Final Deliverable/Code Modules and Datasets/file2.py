"""
Loads data from a CSV file, and calls the necessary
functions to plot histograms, normal distributions,
boxplots, and histograms with outliers removed fo
specified columns in the pandas DataFrame.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_distributions(data: pd.DataFrame) -> None:
    """
    Plots histograms and normal distributions for the
    specified columns in a pandas DataFrame.
    :param data: pd.DataFrame, the pandas DataFrame
        object containing the data to be plotted.
    :return: None.
    """
    # define column names and corresponding labels
    columns = ["LifeExp", "GDP", "CO2", "Health_Expenditure"]
    labels = ["Life Expectancy", "GDP", "CO2", "Health Expenditure"]

    # loop over columns to plot histograms and normal distributions
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, col in enumerate(columns):
        ax = axs[i // 2, i % 2]
        x = data[col]
        mean, std = np.mean(x), np.std(x)
        ax.hist(x, bins=10, density=True)
        fit = norm.pdf(np.linspace(np.min(x), np.max(x), 100), mean, std)
        ax.plot(np.linspace(np.min(x), np.max(x), 100), fit, '-o')
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {labels[i]}")

    plt.tight_layout()
    plt.show()


def no_outliers_plot_distribution(data: pd.DataFrame, col_name: str) -> None:
    """
    Plots a histogram and normal distribution for a
    specified column in a pandas DataFrame after removing outliers.
    :param data: pd.DataFrame, the pandas DataFrame
        object containing the data to be plotted.
    :param col_name: str, the name of the column to be plotted.
    :return: None.
    """
    # calculate quartiles and interquartile range
    q1, q3 = np.percentile(data[col_name], [25, 75])
    iqr = q3 - q1

    # calculate lower and upper bounds for outliers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # filter data to remove outliers
    data_no_outliers = data[(data[col_name] >= lower_bound) &
                            (data[col_name] <= upper_bound)]

    # calculate mean and standard deviation of filtered data
    mean_no_outliers = np.mean(data_no_outliers[col_name])
    std_no_outliers = np.std(data_no_outliers[col_name])

    # plot histogram and normal distribution of filtered data
    plt.hist(data_no_outliers[col_name], bins=10, density=True)
    x = np.linspace(np.min(data_no_outliers[col_name]),
                    np.max(data_no_outliers[col_name]), 100)
    fit = norm.pdf(x, mean_no_outliers, std_no_outliers)
    plt.plot(x, fit, '-o')
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.show()


def plot_boxplot(df: pd.DataFrame, column: str) -> None:
    """
    Plots a boxplot for a specified column in a pandas DataFrame.
    :param df: pd.DataFrame, the pandas DataFrame object
        containing the data to be plotted.
    :param column: str, the name of the column to be plotted.
    :return: None.
    """
    plt.boxplot(df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


def main():
    # Load data
    data = pd.read_csv("clean_data.csv")

    plot_distributions(data)

    no_outliers_plot_distribution(data, "LifeExp")
    no_outliers_plot_distribution(data, "Health_Expenditure")

    plot_boxplot(data, "LifeExp")
    plot_boxplot(data, "Health_Expenditure")


if __name__ == "__main__":
    main()
