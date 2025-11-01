# RealWaste Image Classification (PyTorch)

A deep learning project for classifying waste images using a Convolutional Neural Network (CNN) in PyTorch. This project uses the [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) dataset from the UCI Machine Learning Repository and demonstrates data preprocessing, augmentation, model training, evaluation, and visualization.

## Dataset
- **Source:** [UCI RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste)
- **Classes:** Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation
- **Preprocessing:** Images are split into train/val/test folders and augmented to balance class counts.

## Setup
1. **Clone the repository:**
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
