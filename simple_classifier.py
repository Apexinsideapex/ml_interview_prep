import numpy as np

def train_classifier(X_train, y_train):
    """
    Train a simple classifier to predict player behavior based on game stats.
    
    :param X_train: numpy array of shape (n_samples, n_features) containing training data
    :param y_train: numpy array of shape (n_samples,) containing target values
    :return: dict containing trained model parameters
    """
    # TODO: Implement a simple logistic regression classifier
    # 1. Initialize model parameters (weights and bias)
    # 2. Implement the sigmoid function
    # 3. Implement forward pass
    # 4. Implement backward pass (gradient computation)
    # 5. Update parameters using gradient descent
    # 6. Return the trained parameters
    
    model = {
        'weights': np.zeros(X_train.shape[1]),
        'bias': 0
    }
    return model

def predict(model, X_test):
    """
    Make predictions using the trained classifier.
    
    :param model: dict containing trained model parameters
    :param X_test: numpy array of shape (n_samples, n_features) containing test data
    :return: numpy array of shape (n_samples,) containing predicted probabilities
    """
    # TODO: Implement prediction logic
    # 1. Implement forward pass using the trained parameters
    # 2. Return the predicted probabilities
    
    y_pred = np.zeros(X_test.shape[0])  # Replace with actual predictions
    return y_pred

# Example usage (not needed in the interview):
# X_train = np.random.rand(100, 5)  # 100 samples, 5 features
# y_train = np.random.randint(2, size=100)  # Binary labels
# model = train_classifier(X_train, y_train)
# X_test = np.random.rand(20, 5)  # 20 test samples
# predictions = predict(model, X_test)
# print(predictions)