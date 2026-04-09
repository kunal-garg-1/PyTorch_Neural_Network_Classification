# PyTorch Neural Network Classification 🧠

This repository contains a comprehensive Google Collab Notebook (`02_pytorch_neural_network_classification.ipynb`) dedicated to understanding and implementing **Neural Network Classification** using PyTorch. 

Whether classifying data into two categories (binary) or multiple categories (multi-class), this project breaks down the entire deep learning pipeline—from dataset generation to model evaluation and visualization.

## 📌 Overview

Classification is one of the most foundational problems in machine learning. This notebook demonstrates how to build neural network classification models from scratch, exploring the limitations of strictly linear models and showing how non-linear activation functions help neural networks learn highly complex patterns.

### Key Concepts Covered:
- **Neural Network Architecture:** Understanding input/output shapes, hidden layers, and how to construct a model by subclassing `nn.Module`.
- **Binary Classification:** - Generating a synthetic 2D dataset (e.g., using scikit-learn's `make_circles`).
  - Building a PyTorch model to classify two distinct categories.
  - Applying `BCEWithLogitsLoss` (Binary Cross Entropy).
- **Multi-Class Classification:**
  - Generating data with more than two classes (e.g., using `make_blobs`).
  - Adjusting the network's final layer to output multiple raw logits.
  - Applying `CrossEntropyLoss`.
- **The Power of Non-Linearity:** Implementing activation functions like ReLU (Rectified Linear Unit) alongside Sigmoid and Softmax to map non-linear data sets.
- **The PyTorch Training Loop:** Writing the standard PyTorch workflow (forward pass, calculating loss, zeroing gradients, backpropagation, and stepping the optimizer).
- **Evaluation & Visualization:** Writing helper functions to calculate accuracy and plotting decision boundaries to visually verify what the model has learned.

## 🛠️ Prerequisites & Setup

To run the notebook locally, you will need Python 3.x installed along with the following primary libraries. You can install them via pip:

```bash
pip install torch torchvision torchaudio
pip install matplotlib scikit-learn pandas numpy
