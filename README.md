# Simple pytorch code to test distributed data parallel

# MNIST Training with PyTorch and DDP

This project demonstrates how to train a neural network on the MNIST dataset using PyTorch with the Distributed Data Parallel (DDP) framework. DDP enables efficient parallel training across multiple GPUs, potentially reducing training time and allowing for scalability.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later
- PyTorch 1.8 or later
- torchvision

You can install PyTorch and torchvision by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/yourusername/mnist-pytorch-ddp.git
cd mnist-pytorch-ddp
```

## Usage

The training script can be run on a single machine with multiple GPUs using either of the following methods:

### Method 1: Direct Execution

Simply run the script using Python:

```bash
python train_mnist_ddp.py
```

This method uses torch.multiprocessing.spawn to launch multiple processes for training.

### Method 2: Using torchrun

For a more streamlined approach, especially when scaling to multiple nodes, you can use `torchrun`:

```bash
torchrun --nproc_per_node=<number_of_gpus> train_mnist_ddp.py
```

## Model

The script trains a simple convolutional neural network on the MNIST dataset. The model consists of two convolutional layers followed by two fully connected layers. It uses the Adam optimizer and trains for 10 epochs.
