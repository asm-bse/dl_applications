# Brain Tumor MRI Classification using Deep Convolutional Neural Networks

**Project for the "Applications of Deep Learning" course, Masters in Data Science for Decision Making, Barcelona School of Economics**

**Authors:** Simon Vellin, Aleksandr Smolin, Nicholas Rauth

## 1. Research Question

Brain tumors are among the most serious medical conditions requiring rapid and accurate diagnosis for effective treatment planning. Early detection and precise classification of tumor types can significantly impact patient outcomes. This project aims to address the following research question:

> How accurately can different deep convolutional neural network (CNN) architectures classify brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary tumors?

## 2. Background and Motivation

Recent advancements in deep learning have shown immense potential in medical image diagnostics. Previous studies have successfully applied CNN models for various medical imaging tasks, including tumor detection and classification.

This project extends these approaches by conducting a comprehensive comparative study of three different CNN architectures:
1. **Baseline CNN** - A from-scratch 5-layer convolutional network
2. **VGG19 Transfer Learning** - Leveraging pre-trained ImageNet features
3. **EfficientNet-B0 Transfer Learning** - Modern compound scaling architecture with two-stage training

We utilize advanced training techniques such as data augmentation, batch normalization, class weighting, and adaptive callbacks to prevent overfitting and enhance model accuracy. The goal is to create robust tools that could assist radiologists in making more precise and earlier diagnoses.

## 3. Dataset

### Source and Structure

The project uses the publicly available **"Brain Tumor MRI Dataset"** from Kaggle containing over 7,000 MRI images divided into four balanced classes:
* `Glioma` - A type of brain tumor that occurs in the brain and spinal cord
* `Meningioma` - A tumor that arises from the meninges (membranes surrounding the brain)
* `Notumor` - Healthy brain scans without any tumor presence
* `Pituitary` - Tumors that develop in the pituitary gland

### Preprocessing and Augmentation

1. **Data Splitting:** The data is stratified and split into training (80%) and validation (20%) subsets using `image_dataset_from_directory`. A separate test set is provided by the dataset.
2. **Image Standardization:** All images are resized to 224×224 pixels to ensure compatibility with pre-trained models and consistent input dimensions.
3. **Normalization:** All image pixels are normalized to the `[0, 1]` range using `Rescaling(1./255)`.
4. **Augmentation:** To improve the model's generalization ability and prevent overfitting, the following real-time data augmentation techniques are applied to the training data:
    * Random horizontal flips
    * Random rotations (up to ±10%)
    * Random translations (up to ±5%)
    * Random zooming (up to ±5%)

Model-specific preprocessing is applied for transfer learning models (VGG19 and EfficientNet-B0) to match their pre-training requirements.

## 4. Methodology

### Model Architectures

This project compares three different CNN approaches:

#### Model 1: Baseline CNN (From Scratch)
A deep convolutional neural network designed specifically for medical imaging, consisting of:
* **Five Convolutional Blocks:** Each block contains:
    * `Conv2D` layer with `ReLU` activation and same padding
    * `BatchNormalization` layer to stabilize training
    * `MaxPooling2D` layer to reduce spatial dimensions
* **Classifier Head:**
    * `Flatten` layer to convert 2D feature maps to 1D
    * `Dense` layer (256 neurons) with `ReLU` and `Dropout` (50%)
    * Output `Dense` layer with 4 neurons and `softmax` activation

#### Model 2: VGG19 Transfer Learning
* Pre-trained VGG19 backbone (frozen) from ImageNet
* `GlobalAveragePooling2D` for dimensionality reduction
* Custom classifier: `Dense(256)` + `Dropout(0.5)` + `Softmax(4)`
* Learning rate: 1e-4, trained for 30 epochs

#### Model 3: EfficientNet-B0 Transfer Learning
* Two-stage training approach:
    * Stage 1: Train classifier head only (20 epochs, LR=1e-3)
    * Stage 2: Fine-tune top 20 layers (10 epochs, LR=1e-4)
* Architecture: EfficientNetB0 + `GlobalAveragePooling2D` + `BatchNormalization` + `Dense(512)` + `Dropout(0.3)` + `Dense(256)` + `Dropout(0.3)` + `Softmax(4)`

### Training and Optimization

The training process employs the following techniques:
* **Optimizer:** `Adam` with adaptive learning rates
* **Loss Function:** `categorical_crossentropy` for multi-class classification
* **Metrics:** `accuracy`, `precision`, and `recall` for comprehensive evaluation
* **Class Weighting:** Balanced class weights to address potential data imbalance
* **Callbacks:**
    * **`EarlyStopping`:** Prevents overfitting by monitoring validation metrics
    * **`ModelCheckpoint`:** Saves the best model based on validation accuracy
    * **`ReduceLROnPlateau`:** Adapts learning rate when training plateaus

## 5. Results

The script performs a full evaluation of the best-saved model on the test set. The expected outputs include:
* **Quantitative Metrics:**
    * Test Loss and Test Accuracy for each model.
    * A detailed `classification_report` including Precision, Recall, and F1-score for each class.
    * Statistics on the confidence of the model's predictions (mean, min, max).
* **Visualizations:**
    * **Confusion Matrix:** To visually assess which classes the model confuses most often.
    * **Training History Plots:** Curves for accuracy and loss on both training and validation sets to diagnose overfitting and model convergence.
    * **Model Comparison:** Side-by-side performance comparison of all three architectures.

### Performance Summary
* **Baseline CNN:** ~80% accuracy - Good starting point but struggles with tumor subtype differentiation
* **VGG19 Transfer Learning:** ~88% accuracy - Significant improvement through pre-trained features
* **EfficientNet-B0:** ~95% accuracy - Best performance with two-stage training approach

## 6. How to Run the Script

### Prerequisites

1.  Install the required libraries:
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn kagglehub
    ```
2.  Set up Kaggle API authentication by placing your `kaggle.json` file in the `~/.kaggle/` directory.
### Running the Notebook

1. Open the Jupyter notebook `FINAL_KAGGLE.ipynb`
2. Run all cells sequentially from top to bottom
3. The notebook will automatically:
   - Download the dataset from Kaggle (if not found locally)
   - Perform data exploration and preprocessing
   - Train three different CNN models
   - Generate comparison visualizations and results
   - Save model weights and evaluation metrics

### Output Files

The notebook generates several output files:
- `history_cnn.json`, `history_vgg19.json`, `history_effnetb0.json` - Training histories
- `cm_cnn.npy`, `cm_vgg19.npy`, `cm_effnetb0.npy` - Confusion matrices
- `report_cnn.txt`, `report_vgg19.txt`, `report_effnetb0.txt` - Classification reports
- `advanced_vanilla.h5`, `vgg19_transfer.h5`, `effnetb0_ft.h5` - Trained model weights
