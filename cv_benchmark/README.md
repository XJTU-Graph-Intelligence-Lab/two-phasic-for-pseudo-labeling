
# Non-Stationary Predictions May Be More Informative: Exploring Pseudo-Labels with a Two-Phase Pattern of Training Dynamics

The implementation of all CV benchmarks is based on [USB](https://github.com/microsoft/Semi-supervised-learning.git). It is a Pytorch-based Python package for Semi-Supervised Learning (SSL). You can find the specific running commands in the **"Reproduce CV Benchmark Results"** section.


## Getting Started

This is an example of how to set up USB locally.
To get a local copy up, running follow these simple example steps.

### Prerequisites

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.

### reproduce cv benchmark results

Here is an example to train Two-phase base on Pseudolabel in CIFAR-100 Dataset with 200 labels. Training other supported algorithms (on other datasets with different label settings) can be specified by a config file:

- Step1: Training base model
```
python train.py --c config/two_phase/pseudolabel/cifar_base200.yaml
```
- Step2: Continue training two-phase 
```
python train.py --c config/two_phase/pseudolabel/cifar_tp200.yaml
```

