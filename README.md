# Deep Learning: Batch vs. Stochastic Gradient Descent

This repository provides a deep learning starter-phase demonstration comparing **Batch Gradient Descent** and **Stochastic Gradient Descent (SGD)**. It explores how the choice of `batch_size` during model training impacts learning dynamics, loss convergence, and overall stability.

## Overview
The project uses the `Social_Network_Ads.csv` dataset, predicting whether a user will purchase a product (`Purchased` = 0 or 1) based on two features: `Age` and `EstimatedSalary`[cite: 4]. 
The dataset contains 400 records and is standardized prior to training using scikit-learn's `StandardScaler`[cite: 4].

To compare the different gradient descent techniques, the notebook implements two identical neural network architectures but trains them with different batch sizes.

## Model Architecture
Both models are built using TensorFlow/Keras with the following Sequential architecture[cite: 4]:
*   **Input Layer:** Accepts 2 numerical features (`Age` and `EstimatedSalary`)[cite: 4].
*   **Hidden Layers:** Two consecutive dense layers with 10 neurons each and `relu` activation functions[cite: 4].
*   **Output Layer:** One dense layer with 1 neuron and a `sigmoid` activation for binary classification[cite: 4].
*   **Compilation:** The models are compiled using `binary_crossentropy` loss and accuracy metrics[cite: 4].

## Experiments

### 1. Stochastic Gradient Descent (SGD)
*   The first model is trained using a `batch_size` of 1[cite: 4].
*   This means the model updates its weights after evaluating each individual training sample.
*   It is trained for 500 epochs using a 20% validation split[cite: 4].
*   The training loss history is plotted using matplotlib, typically displaying a noisier and highly variable convergence curve[cite: 4].

### 2. Batch Gradient Descent
*   The second model is trained using a much larger `batch_size` of 250[cite: 4].
*   Since the training set consists of 320 samples (80% of 400), a batch size of 250 processes the vast majority of the data in a single step before updating the network's weights.
*   It is trained for 50 epochs using a 20% validation split[cite: 4].
*   The training loss history is plotted, which reveals a much smoother, more stable descent compared to SGD[cite: 4].

## Dependencies
To run the code in this repository, you will need the following Python libraries installed:
*   `numpy`[cite: 4]
*   `pandas`[cite: 4]
*   `matplotlib`[cite: 4]
*   `tensorflow` (and `keras`)[cite: 4]
*   `scikit-learn`[cite: 4]
