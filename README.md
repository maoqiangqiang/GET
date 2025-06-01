# Gradient-based Entire Tree Optimization For Oblique Decision Tree
This repository has been restructured to offer a more organized and user-friendly interface. GET (Gradient-based Entire Tree) is designed to induce oblique decision trees by optimizing the entire tree structure via gradient-based optimization. It supports both regression and classification tasks. For detailed information on the algorithm, please refer to the study “Can a Single Tree Outperform an Entire Forest?”, available at https://arxiv.org/pdf/2411.17003.


<div align="center">

![Language](https://img.shields.io/badge/language-Python-blue?&logo=python)
![Dependencies Status](https://img.shields.io/badge/dependencies-PyTorch-brightgreen.svg)
<!-- [![License](https://img.shields.io/github/license/maoqiangqiang/GET)](https://github.com/maoqiangqiang/GET/blob/main/LICENSE) -->

</div>


Features in this version:
- `GETRegressor()`: An oblique regression tree with constant predictions.
- `GETSubPolRegressor()`: An oblique regression tree with constant predictions, enhanced with a subtree polishing strategy.

New features will be added in next versions, including:
- tree path-based interpretability
- Classification tree implementations like `GETClassifier()` and `GETSubPolClassifier()`


## Package Dependencies
- scikit-learn 1.5.0
- numpy 1.26.4
- pandas 2.2.3
- h5py 3.13.0
- torch 2.0.0+


## Package Installation
```Shell
pip install get-oblique
```


## Package Description 
`GETRegressor class`: oblique regression tree with constant predictions. <br>
- `Parameters`:
  - `treeDepth` (int, default=4): The depth of the regression tree.
  - `epochNum` (int, default=3000): Number of training epochs used during optimization.
  - `startNum` (int, default=10): Number of random initializations for the tree optimization process (This increases the chance of finding optimal solutions).
  - `device` (str, default='cpu'): The computation device to use: 'cpu' or 'cuda'. Set to 'cuda' to enable GPU acceleration.
- `Methods`:
  - `fit(X, y)`: <br>
    Train the model using gradient-based optimization. Automatically moves data to the specified device and converts to float tensors.
  - `predict(X)`: <br>
    Predicts target values based on trained tree structure.

`GETSubPolRegressor class`: oblique regression tree with constant predictions and subtree polish strategy. <br>
- `Parameters`:
  - `treeDepth` (int, default=4): The depth of the regression tree.
  - `epochNum` (int, default=3000): Number of training epochs used during optimization.
  - `startNum` (int, default=10): Number of random initializations for the tree optimization process (This increases the chance of finding optimal solutions).
  - `device` (str, default='cpu'): The computation device to use: 'cpu' or 'cuda'. Set to 'cuda' to enable GPU acceleration.
- `Methods`:
  - `fit(X, y)`: <br>
    Train the model using gradient-based optimization and subtree polish strategy. Automatically moves data to the specified device and converts to float tensors.
  - `predict(X)`: <br>
    Predicts target values based on trained tree structure.


## Usage Example
To use the GETRegressor class:
```Shell
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from GET import GETRegressor

# Load and prepare dataset
data = fetch_california_housing()

# X, y can be either Numpy arrays or Pytorch tensors, in this case they are numpy arrays
X, y = data.data, data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = GETRegressor()

# Fit the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print sample predictions
print("First 10 predicted values:", y_pred[:10])
```


## Others 
If you encounter any errors or notice unexpected tree performance, please don't hesitate to contact us.

## License
This repository is published under the terms of the `GNU General Public License v3.0 `. 
