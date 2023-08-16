'''
This script reads in several CSV files containing data related to
life expectancy, GDP, health expenditure, and CO2 emissions, cleans
the data by filtering out missing values and merging the data sets,
and saves the cleaned data to a new CSV file called "clean_data.csv".
The cleaned data is structured as a Pandas DataFrame with columns for
country name, year, life expectancy, GDP, health expenditure, and CO2
emissions.
'''

import pandas as pd


def clean_data() -> None:
    """
    Cleans and merges data from multiple CSV files
    and saves it to a new CSV file.
    No parameters are taken.
    Returns nothing.
    """

    def load_data(file_name: str, value_name: str) -> pd.DataFrame:
        """
        Loads data from a CSV file, reshapes it using pandas, and
        returns a pandas DataFrame object.
        Args:
            file_name: str, the name of the CSV file to load.
            value_name: str, the name of the column containing
                           the data values.
        Returns:
            pd.DataFrame: the cleaned pandas DataFrame object.
        """
        data = pd.read_csv(file_name, skiprows=4)
        data = pd.melt(data, id_vars=['Country Name', 'Country Code',
                                      'Indicator Name', 'Indicator Code'],
                       var_name='Year', value_name=value_name)
        data = data[["Country Name", "Country Code", "Year", value_name]]
        data = data.dropna()
        return data

    def merge_data(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Merges multiple pandas DataFrame objects and returns
        the resulting merged DataFrame.

        Args:
        dataframes: list[pd.DataFrame], a list of pandas
                           DataFrame objects to merge.
        Return：
        pd.DataFrame, the resulting merged
                 pandas DataFrame object.
        """
        data = dataframes[0]
        for i in range(1, len(dataframes)):
            data = pd.merge(data, dataframes[i])
        data = data.drop("Country Code", axis=1)
        data = data.replace('', pd.NaT)
        return data

    # Load data
    lifeExp = load_data("Life_Expectancy.csv", "LifeExp")
    gdp = load_data("Gdp_per.csv", "GDP")
    health = load_data("Health_Expenditure.csv", "Health_Expenditure")
    co2 = load_data("CO2_per.csv", "CO2")

    # Merge the data sets
    data = merge_data([lifeExp, gdp, health, co2])

    data.to_csv("clean_data.csv", index=False)


def main():
    clean_data()


if __name__ == "__main__":
    main()
