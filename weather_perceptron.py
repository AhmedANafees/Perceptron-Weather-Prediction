import csv
import random

# =======================================================
# SECTION 1: CONFIGURATION
# =======================================================

# The path to the dataset file.
FILE_PATH = 'WeatherData.csv'

# The number of data points to use for the training set.
# The rest will be used for testing.
TRAIN_SPLIT_SIZE = 15

# The learning rate for the Perceptron model.
LEARNING_RATE = 0.1

# =======================================================
# SECTION 2: HELPER FUNCTIONS
# =======================================================

def load_data_from_csv(filepath):
    """
    Extracts features and labels from a CSV file.
    """
    features = []
    labels = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row

        for row in csv_reader:
            # Convert feature columns (temperature, humidity) to float
            features.append([float(row[0]), float(row[1])])
            # Convert label column (Rain) to integer
            labels.append(int(row[2]))
            
    return features, labels

def calculate_accuracy(y_true, y_pred):
    """
    Calculates prediction accuracy without using external libraries.
    """
    correct_predictions = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_predictions += 1
    return correct_predictions / len(y_true)

# =======================================================
# SECTION 3: PERCEPTRON IMPLEMENTATION
# =======================================================

class Perceptron:
    """
    A Perceptron classifier implementation.
    """
    def __init__(self, learning_rate=LEARNING_RATE, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        """
        Train the Perceptron model.
        """
        n_features = len(X[0])
        
        # Initialize weights randomly between -0.5 and 0.5
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = random.uniform(-0.5, 0.5) # Include a bias term

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                # Calculate linear output (dot product) manually
                linear_output = sum(w * x for w, x in zip(self.weights, x_i)) + self.bias

                # Apply step activation function
                y_predicted = 1 if linear_output >= 0 else 0

                # Calculate update using the learning rate of LEARNING_RATE
                update = self.learning_rate * (y[i] - y_predicted)

                # Update weights and bias
                for j in range(n_features):
                    self.weights[j] += update * x_i[j]
                self.bias += update

    def predict(self, X):
        """
        Predict class labels for a set of input samples.
        """
        predictions = []
        for x_i in X:
            linear_output = sum(w * x for w, x in zip(self.weights, x_i)) + self.bias
            predictions.append(1 if linear_output >= 0 else 0)
        return predictions

# =======================================================
# SECTION 4: MAIN EXECUTION
# =======================================================

def main():
    """
    Main function to load data, train the perceptron, and evaluate its performance.
    """
    # Load the weather dataset
    features, labels = load_data_from_csv(FILE_PATH)

    # Split the dataset into a training set (first TRAIN_SPLIT_SIZE) and a test set (The rest)
    X_train, y_train = features[:TRAIN_SPLIT_SIZE], labels[:TRAIN_SPLIT_SIZE]
    X_test, y_test = features[TRAIN_SPLIT_SIZE:], labels[TRAIN_SPLIT_SIZE:]

    # Instantiate and train the Perceptron model
    perceptron = Perceptron(learning_rate=LEARNING_RATE, n_iters=1000)
    perceptron.fit(X_train, y_train)

    # Report the training and test accuracy after training
    train_predictions = perceptron.predict(X_train)
    test_predictions = perceptron.predict(X_test)

    train_accuracy = calculate_accuracy(y_train, train_predictions)
    test_accuracy = calculate_accuracy(y_test, test_predictions)

    print("--- Perceptron Model Evaluation ---")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Final Parameters:\nWeights: {perceptron.weights}\nBias: {perceptron.bias}")
    
    print("\n--- Plotting Information ---")
    print("To plot the decision boundary, add a line to your graph using these two points:")
    w1, w2 = perceptron.weights[0], perceptron.weights[1]
    bias = perceptron.bias
    # Point 1: Let x1 = 0
    x1_point1 = 0.0
    x2_point1 = (-w1 * x1_point1 - bias) / w2
    # Point 2: Let x1 = 1
    x1_point2 = 1.0
    x2_point2 = (-w1 * x1_point2 - bias) / w2
    print(f"Point 1 (Temperature, Humidity): ({x1_point1:.2f}, {x2_point1:.2f})")
    print(f"Point 2 (Temperature, Humidity): ({x1_point2:.2f}, {x2_point2:.2f})")


if __name__ == "__main__":
    main()
import csv
import random

# =======================================================
# SECTION 1: CONFIGURATION
# =======================================================

# The path to the dataset file.
FILE_PATH = 'WeatherData.csv'

# The number of data points to use for the training set.
# The rest will be used for testing.
TRAIN_SPLIT_SIZE = 15

# The learning rate for the Perceptron model.
LEARNING_RATE = 0.1

# =======================================================
# SECTION 2: HELPER FUNCTIONS
# =======================================================

def load_data_from_csv(filepath):
    """
    Extracts features and labels from a CSV file.
    """
    features = []
    labels = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row

        for row in csv_reader:
            # Convert feature columns (temperature, humidity) to float
            features.append([float(row[0]), float(row[1])])
            # Convert label column (Rain) to integer
            labels.append(int(row[2]))
            
    return features, labels

def calculate_accuracy(y_true, y_pred):
    """
    Calculates prediction accuracy without using external libraries.
    """
    correct_predictions = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_predictions += 1
    return correct_predictions / len(y_true)

# =======================================================
# SECTION 3: PERCEPTRON IMPLEMENTATION
# =======================================================

class Perceptron:
    """
    A Perceptron classifier implementation.
    """
    def __init__(self, learning_rate=LEARNING_RATE, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        """
        Train the Perceptron model.
        """
        n_features = len(X[0])
        
        # Initialize weights randomly between -0.5 and 0.5
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = random.uniform(-0.5, 0.5) # Include a bias term

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                # Calculate linear output (dot product) manually
                linear_output = sum(w * x for w, x in zip(self.weights, x_i)) + self.bias

                # Apply step activation function
                y_predicted = 1 if linear_output >= 0 else 0

                # Calculate update using the learning rate of LEARNING_RATE
                update = self.learning_rate * (y[i] - y_predicted)

                # Update weights and bias
                for j in range(n_features):
                    self.weights[j] += update * x_i[j]
                self.bias += update

    def predict(self, X):
        """
        Predict class labels for a set of input samples.
        """
        predictions = []
        for x_i in X:
            linear_output = sum(w * x for w, x in zip(self.weights, x_i)) + self.bias
            predictions.append(1 if linear_output >= 0 else 0)
        return predictions

# =======================================================
# SECTION 4: MAIN EXECUTION
# =======================================================

def main():
    """
    Main function to load data, train the perceptron, and evaluate its performance.
    """
    # Load the weather dataset
    features, labels = load_data_from_csv(FILE_PATH)

    # Split the dataset into a training set (first TRAIN_SPLIT_SIZE) and a test set (The rest)
    X_train, y_train = features[:TRAIN_SPLIT_SIZE], labels[:TRAIN_SPLIT_SIZE]
    X_test, y_test = features[TRAIN_SPLIT_SIZE:], labels[TRAIN_SPLIT_SIZE:]

    # Instantiate and train the Perceptron model
    perceptron = Perceptron(learning_rate=LEARNING_RATE, n_iters=1000)
    perceptron.fit(X_train, y_train)

    # Report the training and test accuracy after training
    train_predictions = perceptron.predict(X_train)
    test_predictions = perceptron.predict(X_test)

    train_accuracy = calculate_accuracy(y_train, train_predictions)
    test_accuracy = calculate_accuracy(y_test, test_predictions)

    print("--- Perceptron Model Evaluation ---")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Final Parameters:\nWeights: {perceptron.weights}\nBias: {perceptron.bias}")
    
    print("\n--- Plotting Information ---")
    print("To plot the decision boundary, add a line to your graph using these two points:")
    w1, w2 = perceptron.weights[0], perceptron.weights[1]
    bias = perceptron.bias
    # Point 1: Let x1 = 0
    x1_point1 = 0.0
    x2_point1 = (-w1 * x1_point1 - bias) / w2
    # Point 2: Let x1 = 1
    x1_point2 = 1.0
    x2_point2 = (-w1 * x1_point2 - bias) / w2
    print(f"Point 1 (Temperature, Humidity): ({x1_point1:.2f}, {x2_point1:.2f})")
    print(f"Point 2 (Temperature, Humidity): ({x1_point2:.2f}, {x2_point2:.2f})")


if __name__ == "__main__":
    main()
