# Project 3

# ***Interactive CNN-Based CIFAR-10 Classification System using Gradio***

Description: This project focuses on the classification of the CIFAR-10 dataset using Convolutional Neural Networks (CNNs). The primary objective is to develop and evaluate a deep learning model capable of accurately categorizing the 32x32 color images into their respective 10 classes. The methodology involves designing and training a CNN architecture utilizing the TensorFlow/Keras framework, incorporating layers such as Conv2D, MaxPooling2D, BatchNormalization, and Dropout to enhance feature extraction and model generalization. Furthermore, this project emphasizes practical application and model interaction by implementing an interactive web interface using the Gradio library. This interface allows users to upload images directly and receive real-time classification predictions from the trained CNN model. The model's performance is rigorously evaluated using standard metrics, including accuracy and F1-score, on both validation and test sets, demonstrating the effectiveness of the CNN approach for this image classification task. This work provides insights into building effective CNNs for image recognition and highlights the utility of tools like Gradio in creating accessible and interactive machine learning systems.Usage

# Task

* **Classification Task** : Images Classification

# **Ｍodel**

**Five highest model accuracy rankings in the classroom contest**

Accuracy :0.8686

Macro F1 Score : 0.8678

### **CNN Model Architecture**

This project utilizes a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model follows a sequential architecture consisting of three main convolutional blocks followed by a classifier head.

Each convolutional block contains:

* Two `Conv2D` layers with 3x3 kernels and ReLU activation functions, using 'same' padding. The number of filters increases through the blocks (32 -> 64 -> 128) to capture progressively more complex features.
* `BatchNormalization` layers after each convolution to stabilize training.
* A `MaxPooling2D` layer (2x2) for downsampling.
* A `Dropout` layer to prevent overfitting (rates increase from 0.25 to 0.35 across blocks).

The classifier head consists of:

* A `Flatten` layer to convert the feature maps into a 1D vector.
* A `Dense` hidden layer with 256 units and ReLU activation, followed by `BatchNormalization` and `Dropout` (rate 0.5).
* A final `Dense` output layer with 10 units (one for each CIFAR-10 class) using a `softmax` activation function to produce class probabilities.

The model was compiled using the Adam optimizer and categorical crossentropy loss.



##  Class names

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

# Author

Tzu-Chieh Chao
