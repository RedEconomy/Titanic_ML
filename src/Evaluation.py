from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model using accuracy score.

    Arguments:
        model: Trained machine learning model.
        X_test: Features of the test set.
        y_test: True labels of the test set.

    Returns:
        Accuracy score (float).
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Model Accuracy: {accuracy:.2f}")
    return accuracy
