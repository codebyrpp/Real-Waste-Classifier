# RealWaste Image Classification (PyTorch)

A deep learning project for classifying waste images using a Convolutional Neural Network (CNN) in PyTorch. This project uses the [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) dataset from the UCI Machine Learning Repository and demonstrates data preprocessing, augmentation, model training, evaluation, and visualization.

<p align="center">
	<img src="images/sample.png" alt="Dataset Overview" height="600"/><br>
	<em>Figure: Sample image with its label for all 9 classes.</em>
</p>

## Project Structure
```
RealWaste CNN/
├── main.ipynb           # Jupyter notebook with all code
├── README.md            # Project documentation
├── checkpoint.pth       # Model checkpoint
├── history.json         # Training history
├── dataset/             # Original dataset
├── dataset_split/       # Train/val/test splits
```

## Dataset
- **Source:** [UCI RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste)
- **Classes:** Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation
- **Preprocessing:** Images are split into train/val/test folders and augmented to balance class counts.

## Setup
1. **Clone the repository:**
	```bash
	git clone https://github.com/InduwaraGunasena/RealWaste-Image-Classification.git
	cd RealWaste-Image-Classification
	```
2. **Install dependencies:**
	- Python 3.8+
	- PyTorch
	- torchvision
	- scikit-learn
	- matplotlib
	- tqdm
	- pandas
	- seaborn
	- Jupyter Notebook
	```bash
	pip install torch torchvision scikit-learn matplotlib tqdm pandas seaborn jupyter
	```
3. **Download the RealWaste dataset** and place it in the `dataset/realwaste-main/RealWaste` directory.

## Usage
Open `main.ipynb` in Jupyter Notebook and run the cells sequentially:

1. **Environment Setup:**
	- Detects GPU and sets device.
2. **Data Preparation:**
	- Splits dataset into train/val/test.
	- Augments training data for class balance.
	- Optionally caches datasets for faster loading.
3. **Data Visualization:**
	- Shows sample images and class distributions.
4. **Model Definition:**
	- Defines a custom CNN architecture with batch normalization, dropout, and global average pooling.
5. **Training:**
	- Trains the model with progress bars showing loss and accuracy.
	- Supports early stopping and checkpointing.
6. **Evaluation:**
	- Plots training history (loss, accuracy, precision, recall).
	- Evaluates on test set and prints metrics.
	- Displays confusion matrix and classification report.
7. **Prediction Visualization:**
	- Shows grid of test images with predicted/true labels and probability distributions.

## Results & Visualization
- **Training History:** Plots for loss, accuracy, precision, and recall over epochs.
- **Test Predictions:** Visual grids showing model predictions and confidence for test images.
- **Confusion Matrix:** Visualizes class-wise performance.
- **Classification Report:** Detailed metrics for each class.


## Model Architecture & Performance

Our custom CNN model is designed for robust image classification with the following layer-wise structure:

- **Input Layer:** Accepts RGB images of size 128x128.
- **Block 1:** Two convolutional layers (64 filters), each followed by batch normalization and ReLU activation, then max pooling and dropout.
- **Block 2:** Two convolutional layers (128 filters), batch normalization, ReLU, max pooling, and dropout.
- **Block 3:** Two convolutional layers (256 filters), batch normalization, ReLU, max pooling, and dropout.
- **Block 4:** One convolutional layer (256 filters), batch normalization, ReLU, and dropout.
- **Global Average Pooling:** Reduces spatial dimensions to a single value per channel.
- **Fully Connected Layers:**
  - Dense layer (128 units) with ReLU and dropout
  - Output layer (for 9 classes)

This architecture uses batch normalization and dropout throughout to improve generalization and training stability.

**Model Summary:**

The model contains multiple convolutional blocks, global average pooling, and fully connected layers, resulting in a compact and efficient design for multi-class waste classification.

<p align="center">
	<img src="images/model_summary.png" alt="Model Summary" height="600"/><br>
	<em>Figure: Layer-wise summary of the CNN architecture.</em>
</p>

After 50 epochs of training, the model achieved:
- **Accuracy:** 80.47%
- **Precision:** 80.12%
- **Recall:** 81.47%

**Training History:**

The training and validation curves show steady improvement and convergence, with minimal overfitting due to regularization.

<p align="center">
	<img src="images/plots.png" alt="Training History" width="500"/><br>
	<em>Figure: Training and validation metrics over epochs.</em>
</p>

**Confusion Matrix:**

The confusion matrix below visualizes the model's performance across all classes, highlighting areas of strong and weak classification.

<p align="center">
	<img src="images/output.png" alt="Confusion Matrix" width="500"/><br>
	<em>Figure: Confusion matrix for test set predictions.</em>
</p>


## Customization
- Change `batch_size`, `img_size`, or model architecture in `main.ipynb` as needed.
- Adjust augmentation pipeline for different image sizes or transformations.

## Citation
If you use this project or dataset, please cite the UCI RealWaste dataset and this repository.
