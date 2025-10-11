import matplotlib.pyplot as plt


def plot_predictions(X_train, y_train, X_test, y_test, predictions=None) -> plt.figure:
    """
    Plots training data, test data and compares predictions.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()

    ax.scatter(X_train, y_train, c="b", s=4, label="Training data")

    ax.scatter(X_test, y_test, c="g", marker="o", s=6, label="Testing data")

    if predictions is not None:
        ax.scatter(X_test, predictions, c="r", marker="X", s=4, label="Predictions")

    ax.legend(prop={"size": 14})

    return fig
