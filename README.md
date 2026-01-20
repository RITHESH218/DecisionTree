# DecisionTree
Python implementation of the Decision Tree machine learning algorithm


## Overview

This repository contains a clean and efficient implementation of the Decision Tree algorithm, one of the most popular and interpretable machine learning algorithms for classification and regression tasks.

## Features

- Simple and intuitive implementation
- Support for classification and regression tasks
- Configurable splitting criteria (Gini impurity, Information gain)
- Pruning techniques to prevent overfitting
- Easy-to-use API with scikit-learn style interface
- Comprehensive visualization capabilities

## Installation

```bash
git clone https://github.com/RITHESH218/DecisionTree.git
cd DecisionTree
```

## Usage

```python
from decision_tree import DecisionTreeClassifier

# Initialize the model
dt = DecisionTreeClassifier(max_depth=5, criterion='gini')

# Train the model
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)

# Get accuracy
accuracy = dt.score(X_test, y_test)
```

## Requirements

- Python 3.7+
- NumPy
- Pandas (optional)
- Matplotlib (optional, for visualization)
- Scikit-learn (optional, for comparison)

## Algorithm Details

### How Decision Tree Works:

1. **Splitting**: Recursively split the dataset based on feature values that maximize information gain or minimize impurity
2. **Stopping**: Stop when maximum depth is reached, minimum samples per node, or pure nodes are created
3. **Prediction**: Traverse from root to leaf based on feature values and return the majority class (or mean value for regression)

### Splitting Criteria:

- **Gini Impurity**: Measures the probability of a random sample being misclassified
- **Information Gain**: Based on entropy reduction (Shannon information)
- **Variance Reduction**: Used for regression tasks

## Complexity Analysis

- **Time Complexity**: O(n * m * log(n)) for training (n = samples, m = features)
- **Space Complexity**: O(m * log(n)) for tree storage
- **Prediction**: O(log(n)) per sample

## Advantages

✓ Interpretable and easy to understand
✓ Requires no feature scaling
✓ Handles non-linear relationships
✓ Works with categorical and numerical features
✓ Fast prediction time

## Disadvantages

✗ Prone to overfitting
✗ Biased towards features with more levels
✗ Unstable with small data changes
✗ May create unbalanced trees

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Author

RITHESH218

## License

MIT License
