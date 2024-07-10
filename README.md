# Moon-and-Spiral-Data-Classification-with-PyTorch
This project demonstrates the use of PyTorch to build, train, and evaluate neural network models for classifying synthetic moon and spiral datasets. It covers key concepts in binary and multi-class classification, including data preparation, model building, and performance visualization.

## Features
- **Device Agnostic Code:** Automatically uses GPU if available.
- **Random Seed Initialization:** Ensures reproducible results.
- **Data Preparation:**
  - Creates a synthetic moon dataset using make_moons from Scikit-Learn.
  - Generates a spiral dataset, commonly used in CS231n for neural network case studies.
- **Data Visualization:** Utilizes Matplotlib to visualize data and decision boundaries.
- **Model Building:**
  - Binary Classification Model (MoonModelV0): Subclassed from nn.Module.
  - Multi-Class Classification Model (SpiralModel): Subclassed from nn.Module.
- **Training and Testing:**
  - Training loop with loss and accuracy calculation.
  - Evaluation with testing data.
- **Custom Activation Function:** Implementation of the Tanh activation function in pure PyTorch.

## Dependencies
PyTorch
Scikit-Learn
Matplotlib
Pandas
Numpy
TorchMetrics
## How to Run
1. **Install Dependencies:**

```pip install torch scikit-learn matplotlib pandas numpy torchmetrics```

2. **Run the Script:**

```python moon_spiral_classification.py```

3. **View Results:** The script will output training and testing loss/accuracy and plot decision boundaries for both training and test sets.
