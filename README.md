# Image Classifier with Deep Learning

## Project Overview
This project implements an image classification model using deep learning with **PyTorch**. The model is trained on the **Flower Dataset**, and the trained network can predict the class of a given image.

The project consists of two main components:
1. **train.py** - Trains a new deep learning model and saves it as a checkpoint.
2. **predict.py** - Loads a trained model and predicts the class of an input image.

## Features
- Train a deep learning model on an image dataset
- Save and load model checkpoints
- Perform image classification on a new image
- Use command-line arguments to specify model architecture, hyperparameters, and computation device (CPU/GPU)

## Installation
To use this project,  need to have Python and the following libraries installed:

```sh
pip install torch torchvision numpy matplotlib argparse PIL
```

Alternatively, if using **conda**, create an environment and install dependencies:

```sh
conda create --name image_classifier python=3.8
conda activate image_classifier
pip install torch torchvision numpy matplotlib argparse PIL
```

## Dataset
The project uses the **Flowers dataset**. The data is structured as follows:
```
data_directory/
    train/
    valid/
    test/
```
Make sure to download and organize the dataset before training the model.

## Usage
### 1. Training the Model
Run `train.py` with different options:

```sh
python train.py data_directory --save_dir save_directory --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
```
Options:
- `--save_dir` : Directory to save the model checkpoint
- `--arch` : Choose model architecture (e.g., `vgg13`, `resnet18`)
- `--learning_rate` : Learning rate for training
- `--hidden_units` : Number of hidden units in the classifier
- `--epochs` : Number of training epochs
- `--gpu` : Use GPU for training (if available)

### 2. Predicting an Image Class
Run `predict.py` to classify an image:

```sh
python predict.py /path/to/image checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```
Options:
- `--top_k` : Return top K most likely classes
- `--category_names` : Map class indices to real names
- `--gpu` : Use GPU for inference

## Model Checkpoint
The trained model is saved as a `.pth` file. Be cautious with large files (>1GB) to avoid workspace storage issues. Use:
```sh
ls -lh
```
to check file sizes and move them if necessary.

