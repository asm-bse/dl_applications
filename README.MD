# Alzheimer's Disease Stage Classification using Deep Convolutional Neural Networks

**Project for the "Applications of Deep Learning" course, Masters in Data Science for Decision Making, Barcelona School of Economics**

**Authors:** Aleksandr Smolin, Simon Vellin

## 1. Research Question

Alzheimer's disease is a leading cause of dementia worldwide, creating a significant economic and social burden. Early and accurate diagnosis is crucial for planning treatment and improving patients' quality of life. This project aims to address the following research question:

> How accurately can a deep convolutional neural network (CNN) classify the four stages of Alzheimer's disease (from 'Non-Demented' to 'Moderate Demented') using structural MRI scans of the brain?

## 2. Background and Motivation

Recent advancements in deep learning have shown immense potential in medical image diagnostics. Previous studies (e.g., Smith et al., 2018; Johnson & Lee, 2020) have successfully applied basic CNN models for binary classification (demented vs. healthy).

This project extends these approaches by tackling the more complex task of multi-class classification. We utilize a deeper, more modern CNN architecture and employ advanced training techniques such as data augmentation, batch normalization, and adaptive callbacks to prevent overfitting and enhance model accuracy. The goal is to create a robust tool that could assist radiologists in making a more precise and earlier diagnosis.

## 3. Dataset

### Source and Structure

The project uses the publicly available **"Alzheimer's Multiclass Dataset (44k Images)"** from Kaggle. It contains 44,000 MRI images divided into four balanced classes:
* `NonDemented`
* `VeryMildDemented`
* `MildDemented`
* `ModerateDemented`

### Preprocessing and Augmentation

1.  **Data Splitting:** The data is stratified and split into a training set (80%) and a testing set (20%) to preserve class distribution. The training set is further divided into training (80%) and validation (20%) subsets to monitor the training process.
2.  **Augmentation:** To improve the model's generalization ability and prevent overfitting, the following real-time data augmentation techniques are applied to the training data using `ImageDataGenerator`:
    * Random rotations (up to 30 degrees)
    * Random width and height shifts (up to 20%)
    * Horizontal flipping
    * Random zooming (up to 20%)
    * Brightness adjustments

All image pixels are normalized to the `[0, 1]` range.

## 4. Methodology

### Model Architecture

A deep Convolutional Neural Network (CNN) was designed, consisting of several key components:
* **Four Convolutional Blocks:** Each block contains:
    * Two `Conv2D` layers with `ReLU` activation function.
    * A `BatchNormalization` layer after each convolution to stabilize and accelerate training.
    * A `MaxPooling2D` layer to reduce spatial dimensions.
    * A `Dropout` layer (25%) for regularization.
* **Fully Connected Layers:**
    * A `Flatten` layer to convert 2D feature maps into a 1D vector.
    * Two hidden `Dense` layers (512 and 256 neurons) with `ReLU`, `BatchNormalization`, and `Dropout` (50%).
    * An output `Dense` layer with 4 neurons and a `softmax` activation function for multi-class classification.

### Training and Optimization

The training process was controlled using the following techniques:
* **Optimizer:** `Adam` with an initial learning rate of `0.001`.
* **Loss Function:** `categorical_crossentropy`, a standard choice for multi-class classification.
* **Metrics:** `accuracy` and `TopKCategoricalAccuracy` (top-3) for a more comprehensive evaluation.
* **Callbacks:**
    * **`EarlyStopping`:** Stops training if the validation loss (`val_loss`) does not improve for 10 epochs and restores the best model weights.
    * **`ModelCheckpoint`:** Saves the model version that achieves the best validation accuracy (`val_accuracy`) to `best_alzheimer_model.h5`.
    * **`ReduceLROnPlateau`:** Reduces the learning rate by a factor of 5 (`factor=0.2`) if `val_loss` does not improve for 5 epochs, helping the model escape local plateaus.

## 5. Results

The script performs a full evaluation of the best-saved model on the test set. The expected outputs include:
* **Quantitative Metrics:**
    * Test Loss and Test Accuracy.
    * A detailed `classification_report` including Precision, Recall, and F1-score for each class.
    * Statistics on the confidence of the model's predictions (mean, min, max).
* **Visualizations:**
    * **Confusion Matrix:** To visually assess which classes the model confuses most often.
    * **Training History Plots:** Curves for accuracy and loss on both training and validation sets to diagnose overfitting and model convergence.

## 6. How to Run the Script

### Prerequisites

1.  Install the required libraries:
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn kagglehub
    ```
2.  Set up Kaggle API authentication by placing your `kaggle.json` file in the `~/.kaggle/` directory.
3.  For Apple Silicon users, ensure the `tensorflow-metal` plugin is installed:
    ```bash
    pip install tensorflow-metal
    ```
