"""
Loads data from a CSV file, and calls the necessary functions
to create scatter plots,
country-specific scatter plots, and heatmaps to visualize
correlations between columns in the pandas DataFrame.
"""
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_scatter_plot(data: pd.DataFrame, x: str, y: str) -> None:
    """
    Creates a scatter plot with a trendline for the specified
    x and y columns in a pandas DataFrame.
    :param data: pd.DataFrame, the pandas DataFrame object
        containing the data to be plotted.
    :param x: str, the name of the column
        to be plotted on the x-axis.
    :param y: str, the name of the column
        to be plotted on the y-axis.
    :return: None.
    """
    scatter_plot = px.scatter(data, x=x, y=y, trendline="ols")
    scatter_plot.show()


def create_country_scatter_plot_to(data: pd.DataFrame) -> None:
    """
    Creates a scatter plot with trendlines for the Life
    Expectancy vs Health Expenditure relationship
    for four countries (United States, China, Germany,
    South Africa) in a pandas DataFrame.
    :param data: pd.DataFrame, the pandas DataFrame
        object containing the data to be plotted.
    :return: None.
    """
    fourCountries_data = data[data["Country Name"].isin(["United States",
                              "China", "Germany", "South Africa"])]
    scatter_plot = px.scatter(fourCountries_data, x="Health_Expenditure",
                              y="LifeExp", color='Country Name',
                              trendline="ols")
    scatter_plot.show()


def plot_corr_heatmap(data: pd.DataFrame, drop_cols: list[str] = [],
                      annot: bool = True, cmap: str = 'coolwarm',
                      title: str = '') -> None:
    """
    Plots a heatmap to visualize the correlation matrix of
    the specified columns in a pandas DataFrame.
    :param data: pd.DataFrame, the pandas DataFrame objec
     containing the data to be plotted.
    :param drop_cols: list[str], a list of columns to drop
        before plotting the heatmap (default=[]).
    :param annot: bool, a flag indicating whether to display
        the correlation coefficients on the heatmap (default=True).
    :param cmap: str, the name of the matplotlib colormap to use
        for the heatmap (default='coolwarm').
    :param title: str, the title of the heatmap (default='').
    :return: None.
    """
    data = data.drop(columns=drop_cols)
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap)
    plt.title(title)
    plt.show()


def main():
    data = pd.read_csv("clean_data.csv")
    create_scatter_plot(data, 'Health_Expenditure', 'LifeExp')
    create_country_scatter_plot_to(data)
    create_scatter_plot(data, 'CO2', 'LifeExp')
    create_scatter_plot(data, 'GDP', 'LifeExp')
    plot_corr_heatmap(data, drop_cols=['Year'], annot=True, cmap='rocket_r',
                      title='heatmap')


if __name__ == "__main__":
    main()
