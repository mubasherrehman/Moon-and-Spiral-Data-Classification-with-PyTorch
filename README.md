# Moon-and-Spiral-Data-Classification-with-PyTorch
This project demonstrates the use of PyTorch to build, train, and evaluate neural network models for classifying synthetic moon and spiral datasets. It covers key concepts in binary and multi-class classification, including data preparation, model building, and performance visualization.
![Moon classificaiton](https://github.com/mubasherrehman/Moon-and-Spiral-Data-Classification-with-PyTorch/assets/73284490/4c112ccf-0acc-401f-a2c4-9b7d3e48d6ff)

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
  - PyTorch
  - Scikit-Learn
  - Matplotlib
  - Pandas
  - Numpy
  - TorchMetrics

## How to Run
1. **Install Dependencies:**

```pip install torch scikit-learn matplotlib pandas numpy torchmetrics```

2. **Run the Script:**

```python moon_spiral_classification.py```

3. **View Results:** The script will output training and testing loss/accuracy and plot decision boundaries for both training and test sets.

## Example Output
The model's performance and decision boundaries will be visualized as follows:
  - Training and Testing Accuracy: Logged every 100 epochs.
  - Decision Boundary Plots: Displayed for both moon and spiral datasets.
![Spiral classification](https://github.com/mubasherrehman/Moon-and-Spiral-Data-Classification-with-PyTorch/assets/73284490/2c6842e6-8971-4b92-946c-e09f11ab27aa)
# Code Overview
**Moon Dataset**
- **Data Preparation:** Generate and visualize the moon dataset.
- **Model Building:** Define and instantiate a binary classification model.
- **Training Loop:** Train the model and evaluate performance.
- **Plot Results:** Visualize the decision boundaries.
**Spiral Dataset**
- **Data Preparation:** Generate and visualize the spiral dataset.
- **Model Building:** Define and instantiate a multi-class classification model.
- **Training Loop:** Train the model and evaluate performance.
- **Plot Results:** Visualize the decision boundaries.

## Custom Activation Function
Implementation of Tanh activation function from scratch and comparison with ```torch.tanh()```.
