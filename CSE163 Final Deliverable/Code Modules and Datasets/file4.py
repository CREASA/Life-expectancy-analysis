'''
The code defines several functions, including plot_tree,
classify_lifeexp, plot_depth_score, plot_feature_importances,
and decision_tree.The file4 performs decision tree analysis
on the input data to predict the Life Expectancy of different
age groups based on GDP, CO2 emissions, and Health Expenditure.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as dtc


def plot_tree(model: dtc, features: list[str], labels: list[str]) -> None:
    """
    Plot the decision tree graph using Graphviz.

    Args:
        model (DecisionTreeClassifier): The decision tree model.
        features (list): The list of feature names.
        labels (list): The list of target label names.

    Returns:
        None
    """
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=features,
                               class_names=labels,
                               impurity=False,
                               filled=True, rounded=True,
                               special_characters=True)
    graphviz.Source(dot_data).render('tree.gv', format='png')
    # display(Image(filename='tree.gv.png'))


def classify_lifeexp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify life expectancy into different age groups.

    Args:
        df (DataFrame): The input DataFrame containing LifeExp column.

    Returns:
        A new DataFrame with LifeExp column classified into different
        age groups.
    """
    for i in df.LifeExp.values:
        if i < 50:
            df.LifeExp.replace(i, 'less than 50', inplace=True)
        elif i < 60:
            df.LifeExp.replace(i, '50~60', inplace=True)
        elif i < 70:
            df.LifeExp.replace(i, '60~70', inplace=True)
        elif i < 80:
            df.LifeExp.replace(i, '70~80', inplace=True)
        else:
            df.LifeExp.replace(i, 'greater than 80', inplace=True)
    return df


def plot_depth_score(scores: 'list[float]') -> None:
    """
    Plot the scores against different max_depth values.

    Args:
        scores (list): A list of accuracy scores.

    Returns:
        None
    """
    plt.plot(range(1, len(scores)+1), scores, color='red')
    plt.ylabel('score')
    plt.xlabel('max_depth')
    plt.xticks(range(1, len(scores)+1))
    plt.savefig('max_depth_score.png')
    plt.clf()


def plot_feature_importances(feature_importances: np.ndarray,
                             feature_names: list[str]) -> None:
    """
    Plot the feature importances against different max_depth values.

    Args:
        feature_importances (ndarray): A numpy array containing
        feature importances.
        feature_names (list): A list of feature names.

    Returns:
        None
    """
    feature_importances = np.array(feature_importances)
    row, col = feature_importances.shape
    for i in range(col):
        plt.plot(range(1, row+1), feature_importances[:, i])
    plt.ylabel('weight')
    plt.xlabel('max_depth')
    plt.xticks(range(1, row+1))
    plt.legend(feature_names)
    plt.savefig('feature_impotance.png')
    plt.clf()


def decision_tree(df: pd.DataFrame, independent_vars: list[str],
                  dependent_var: str) -> None:
    """
    Train and test a decision tree model and plot the results.

    Args:
        df (DataFrame): The input DataFrame.
        independent_vars (list): A list of independent variable names.
        dependent_var (str): The dependent variable name.

    Returns:
        None
    """
    X_var = df[independent_vars].values
    y_var = df[dependent_var].values

    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var,
                                                        test_size=0.2,
                                                        random_state=0)

    # Create two decision tree classifiers,
    # one with a max depth of 3 and one without
    clf1 = dtc(criterion='entropy', max_depth=3, random_state=10)
    clf2 = dtc(criterion='entropy', random_state=10)
    # Train the classifiers on the training data
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    # Use the trained classifiers to predict the labels of the test data
    y_pred1 = clf1.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    # Print the accuracy of the predictions
    print('Accuracy is(max_depth is set to 3):',
          accuracy_score(y_test, y_pred1))
    print('Accuracy is(max_depth is not set):',
          accuracy_score(y_test, y_pred2))

    # Get the feature names and target names for plotting the decision trees
    # and feature importance graphs
    feature_names = independent_vars
    target_names = df[dependent_var].unique().tolist()
    # Plot the decision tree of the classifier with max depth of 3
    plot_tree(clf1, feature_names, target_names)

    # Calculate and print the feature importance of both classifiers
    model1_feature_importance = map(lambda x: round(x, 2),
                                    clf1.feature_importances_)
    model2_feature_importance = map(lambda x: round(x, 2),
                                    clf2.feature_importances_)

    print('Feature importance(max_depth is set to 3):',
          list(zip(feature_names, model1_feature_importance)))
    print('Feature importance(max_depth is not set):',
          list(zip(feature_names, model2_feature_importance)))

    # Calculate the accuracy scores and feature importances of classifiers
    # with different max depths
    scores = []
    feature_importances = []

    for i in range(12):
        clf = dtc(criterion='entropy', max_depth=i+1, random_state=10)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        feature_importances.append(clf.feature_importances_)

    # Plot the accuracy scores and feature importances
    plot_depth_score(scores)
    plot_feature_importances(feature_importances, feature_names)


def main():
    data = pd.read_csv('clean_data.csv')

    df = classify_lifeexp(data)
    df = df[df['Year'] > 2010]
    print('-----------------------------------')
    decision_tree(df, ['GDP', 'CO2', 'Health_Expenditure'], 'LifeExp')


if __name__ == '__main__':
    main()
