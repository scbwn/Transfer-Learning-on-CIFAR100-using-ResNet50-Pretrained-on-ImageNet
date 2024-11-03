# Transfer-Learning-on-CIFAR100-using-ResNet50-Pretrained-on-ImageNet

## Project Overview

This project demonstrates the application of transfer learning on the CIFAR100 dataset using a ResNet50 model pretrained on ImageNet. The goal is to leverage the knowledge learned from large-scale ImageNet dataset and adapt it to the CIFAR100 dataset for improved performance.

## Features

- Utilizes ResNet50 architecture pretrained on ImageNet
- Fine-tuning of pretrained model on CIFAR100 dataset

## Implementation Details

- Framework: TensorFlow 2.x
- Dataset: CIFAR100 (60,000 32x32 color images)
- Pretrained Model: ResNet50 on ImageNet
- Fine-tuning: Preprocess input, batch normalization, and dropout
- Optimization: Adam optimizer with small learning rate

## Requirements

- TensorFlow 2.x
- NumPy

## Usage

1. Clone repository
2. Install requirements
3. Download CIFAR100 dataset
4. Run training script
5. Evaluate model performance on test dataset

## Example Use Cases

- Transfer learning for image classification tasks
- Leveraging pretrained models for improved performance
- Fine-tuning techniques for adapting to new datasets
