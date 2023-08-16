'''
The file defines two functions named P1_Model and P2_Model,
each taking a Pandas DataFrame as input and returning
the summary of an OLS model. The file3 performs
linear regression analysis on the input data to
predict Life Expectancy.
'''


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def P1_Model(data: pd.DataFrame) -> str:
    """
    This function takes a Pandas DataFrame data as input. It fits
    an ordinary least squares(OLS) linear regression model to the
    data, with LifeExp as the dependent variable(y) and Health
    Expenditure as the independent variable(x). This function prints
    the predicted Life Expectancy values based on Health Expenditure,
    and the root mean squared error of the model on the training set,
    and returns the summary of OLS Model.
    """
    GDP = data

    y = GDP.LifeExp
    X = GDP[['Health_Expenditure']]

    m1 = smf.ols("LifeExp ~ Health_Expenditure", data=GDP).fit()
    # predict life expectancy under all Health Expenditure
    prediction = m1.predict(X)
    print('P1_Model Prediction:')
    print(prediction)
    rmse = np.sqrt(np.mean((y - prediction) ** 2))
    print('P1_Model rmse:')
    print(rmse)
    adjusted_mse =\
        rmse / (GDP.LifeExp.max() - GDP.LifeExp.min())
    print('Adjusted Mse:')
    print(adjusted_mse)
    GDP['Prediction'] = m1.predict(GDP.Health_Expenditure)
    plt.plot(GDP.Health_Expenditure, GDP.Prediction)
    plt.title('Life Expectancy Prediction')
    plt.xlabel('Health Expenditure')
    plt.ylabel('Life Expectancy')
    plt.savefig('Prediction_Graph.png')

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)
    m = LinearRegression()
    m_test = m.fit(Xt, yt)
    yhat = m_test.predict(Xt)
    rmset = np.sqrt(np.mean((yt - yhat)**2))
    yhat = m_test.predict(Xv)
    rmsev = np.sqrt(np.mean((yv - yhat)**2))
    print('P1_Model Test Score:')
    print(rmset, rmsev)

    return m1.summary()


def P2_Model(data: pd.DataFrame) -> str:
    """
    This function takes a Pandas DataFrame data as input. It fits
    an ordinary least squares(OLS) linear regression model to the
    data, with LifeExp as the dependent variable(y) and GDP,
    CO2 emission as the independent variable(x). This function prints
    the predicted Life Expectancy values based on GDP and CO2 emission,
    and the root mean squared error of the model on the training set,
    and returns the summary of OLS Model.
    """
    # Generate new columns used for creating model
    new_gdp = data

    # GDP level model
    m_level = smf.ols("LifeExp ~ CO2 + GDP_Level", data=new_gdp).fit()
    print(m_level.summary())

    # Linear Regression Model based on GDP
    X = new_gdp[['GDP', 'CO2']]
    y = new_gdp.LifeExp
    m2 = smf.ols('LifeExp ~ GDP + CO2', data=new_gdp).fit()
    prediction = m2.predict(X)
    print('P2_Model Prediction:')
    print(prediction)
    rmse = np.sqrt(np.mean((y - prediction) ** 2))
    print('P2_Model RMSE:')
    print(rmse)
    adjusted_mse =\
        rmse / (new_gdp.LifeExp.max() - new_gdp.LifeExp.min())
    print('Adjusted Mse:')
    print(adjusted_mse)

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2)
    m = LinearRegression()
    m_test = m.fit(Xv, yv)
    yhat = m_test.predict(Xt)
    rmset = np.sqrt(np.mean((yt - yhat)**2))
    yhat = m_test.predict(Xv)
    rmsev = np.sqrt(np.mean((yv - yhat)**2))
    print('P2_Model Test Score:')
    print(rmset, rmsev)

    return m2.summary()


def main():
    data = pd.read_csv('clean_data.csv')
    print(P1_Model(data))

    df = pd.read_csv('clean_data.csv')

    new_gdp = df.sort_values(['Country Name', 'Year'])
    new_gdp['PrevGDP'] = new_gdp.groupby('Country Name')['GDP'].shift(1)
    new_gdp['GDP_Growth'] =\
        (new_gdp['GDP'] - new_gdp['PrevGDP']) / new_gdp['PrevGDP']
    high_level = new_gdp['GDP_Growth'] >= 0.02
    new_gdp['GDP_Level'] = np.where(high_level == 'if fond is True',
                                    'high_gdp',
                                    'low_gdp')
    new_gdp['GDP_Level_num'] = np.where(high_level == 'if fond is True', 1, 0)

    print(P2_Model(new_gdp))


if __name__ == '__main__':
    main()
