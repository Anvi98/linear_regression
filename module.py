import matplotlib.pyplot as plt

### Splitting for training and testing
def dataSplitter(X, Y, percent):
    X_train = X[:int(len(X)*percent)]
    X_test = X[int(len(X)*percent):]
    Y_train = Y[:int(len(Y)*percent)]
    Y_test = Y[int(len(Y)*percent):]

    return X_train, Y_train, X_test, Y_test

def plot_predictions(train_data, train_labels, test_data,
                    test_labels,
                    predictions = None):
    """
    Plot training data, test data and compares predictions.
    """
    plt.figure(figsize = (10, 7))

    # plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label="training_data")

    # plot test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label="testing_data")

    # Are there predictions?
    if predictions is not None:
    # Plot predictions
        plt.scatter(test_data, predictions, c='r', label="predictions")

    plt.legend(prop={"size": 14})
    plt.show()