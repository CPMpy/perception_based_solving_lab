# Perception-based solving with CPMpy

In this lab session you will get to play with the data from the visual [Sudoku Assistant](https://sudoku-assistant.cs.kuleuven.be). Given a (well-centered) image of a sudoku grid, you will train neural networks to predict the digits, you will develop a constraint model for sudoku solving, and you will link the prediction (probabilities) to the solver to get a CP-based joint inference method that can do better than simply solving for the predicted values. The lab session will be interactive, with open ended questions that require experimenting with the provided Jupyter Python Notebooks, as well as an open part at the end of things that can be further improved (different options requiring different skill levels).

## Installation instructions

### Conda, recommended

The most convenient way of running notebooks locally is to install requires packages within a virtual environement, using conda. This way, scripts and libraries installed for this lab are isolated from others on your machine.
First, make sure that conda is installed. Refer to [this guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it on your system.

After cloning this repository from github, create a new virtual environment that installs the neede packages locally, with the following command:

```bash
conda env create -n ml4cp --file environment.yml
```

Then to activate this environment, use
```bash
conda activate ml4cp
```
Or, on Windows
```
source activate ml4cp
```

You can install any additional library using conda or pip.

### Pip

Alternatively, instead of using conda, you can also install [Python 3.9](https://www.python.org/downloads/release/python-390/) and all required packages with pip

```bash
pip install -r requirements.txt
```

If this fails on linux due to cuda-specific versions of pytorch, you can find the [pytorch exact version installation commands here](https://pytorch.org/get-started/previous-versions/#v1100), such as <i>pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html</i>

## Running notebooks

After install the packages using conda or pip, you can run open notebooks by launching the (local browser-based) jupyter notebook system:

```bash
jupyter notebook
```

## Outline

### Part I: Introduction

This notebook checks that all required software install and run properly. It also download and extract the dataset used for this tutorial [[Notebook](notebooks/01_introduction/0_check_setup.ipynb)]

### Part II: Neural Network Training

This notebook guides you through training a CNN to recognize Sudoku digits from smartphone pictures [[Notebook](notebooks/02_neural_network/neural_net_training.ipynb)]

### Part III: Perception-based Constraint Solving

In the third part, you will model the Visual Sudoku problem with CPMpy, and integrate the probabilistic output of your CNN to search a maximum-likelihood solution.
 [[Notebook](notebooks/03_perception_based_constraint_solving/perception_based_constraint_solving.ipynb)]

### Part IV: Your Turn

You will find advanced challenge at the end of Part II and part III. These are open ended problems you can explore after the tutorial.

- Unique solutions (Nogood) [CP]
- Handling data imbalance [ML]
- Handwritten and Printed values [ML and CP]
